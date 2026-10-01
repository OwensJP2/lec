"use strict";

const $ = (id) => document.getElementById(id);
const state = { output: "", mode: "encode", outputMode: "encode" };

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    state.mode = tab.dataset.mode;
    document.querySelectorAll(".tab").forEach((item) => {
      const active = item === tab;
      item.classList.toggle("active", active);
      item.setAttribute("aria-selected", String(active));
    });
    ["encode", "decode"].forEach((mode) => {
      const panel = $(`${mode}-panel`);
      const active = mode === state.mode;
      panel.hidden = !active;
      panel.classList.toggle("active", active);
    });
    $("error").hidden = true;
    $("result").hidden = true;
  });
});

$("message").addEventListener("input", () => {
  $("message-bytes").textContent = String(new TextEncoder().encode($("message").value).length);
});

async function configureEnvironment() {
  try {
    const response = await fetch("/api/status");
    const status = await response.json();
    if (status.real_model_available === false) {
      ["encode", "decode"].forEach((mode) => {
        const toggle = $(`${mode}-toy`);
        toggle.checked = true;
        toggle.disabled = true;
        $(`${mode}-model-title`).textContent = "Hosted demo model";
        $(`${mode}-model-help`).textContent =
          "Vercel uses the lightweight deterministic model; run locally for DistilGPT-2";
      });
    }
  } catch {
    // The forms will surface a useful API error if status cannot be loaded.
  }
}

async function request(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error(`Server returned an invalid response (${response.status}).`);
  }
  if (!response.ok) {
    const detail = typeof data.detail === "string" ? data.detail : "Request failed.";
    throw new Error(detail);
  }
  return data;
}

function setBusy(form, busy, mode) {
  const button = form.querySelector(".primary");
  button.disabled = busy;
  button.textContent = busy
    ? (mode === "encode" ? "Encoding…" : "Decoding…")
    : (mode === "encode" ? "Encode message" : "Decode covertext");
}

function showResult(data, mode) {
  state.outputMode = mode;
  state.output = mode === "encode" ? data.text : data.message;
  $("result-label").textContent = mode === "encode" ? "COVERTEXT" : "RECOVERED MESSAGE";
  $("result-title").textContent = mode === "encode" ? "Generated text" : "Decoded text";
  // User/API content is always assigned as text, never interpreted as markup.
  $("result-output").textContent = state.output;
  $("metric-tokens").textContent = String(data.metrics.tokens);
  $("metric-payload").textContent = String(data.metrics.payload_bits);
  $("metric-bpt").textContent = Number(data.metrics.bits_per_token).toFixed(2);
  $("metric-entropy").textContent =
    `${(Number(data.metrics.entropy_utilization) * 100).toFixed(1)}%`;
  $("result-status").textContent = mode === "encode" ? "Generation complete" : "Recovery complete";
  $("result-status").classList.remove("live");
  $("result").hidden = false;
  $("result").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function codecOptions(mode) {
  return {
    capacity: Number($(`${mode}-capacity`).value),
    toy: $(`${mode}-toy`).checked,
    temperature: Number($(`${mode}-temperature`).value),
    top_k: Number($(`${mode}-top-k`).value),
    top_p: Number($(`${mode}-top-p`).value),
    precision_bits: Number($(`${mode}-precision`).value),
    repetition_penalty: Number($(`${mode}-repetition`).value),
    no_repeat_ngram: Number($(`${mode}-ngram`).value),
  };
}

function copyEncodeConfiguration() {
  ["capacity", "temperature", "top-k", "top-p", "precision", "repetition", "ngram"].forEach((name) => {
    $(`decode-${name}`).value = $(`encode-${name}`).value;
  });
  $("decode-toy").checked = $("encode-toy").checked;
}

function beginStream() {
  state.outputMode = "encode";
  state.output = "";
  $("result-label").textContent = "LIVE COVERTEXT";
  $("result-title").textContent = "Generating text";
  $("result-output").textContent = "";
  $("metric-tokens").textContent = "0";
  $("metric-payload").textContent = "—";
  $("metric-bpt").textContent = "—";
  $("metric-entropy").textContent = "—";
  $("result-status").textContent = "Waiting for the first model token…";
  $("result-status").classList.add("live");
  $("result").hidden = false;
  $("result").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function streamEncode(body) {
  const response = await fetch("/api/encode/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    let detail = `Request failed (${response.status}).`;
    try {
      const data = await response.json();
      if (typeof data.detail === "string") detail = data.detail;
    } catch { /* retain the HTTP error */ }
    throw new Error(detail);
  }
  if (!response.body) throw new Error("Streaming is not supported by this browser.");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let completed = false;
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (!line.trim()) continue;
      const event = JSON.parse(line);
      if (event.type === "token") {
        state.output = event.text;
        $("result-output").textContent = event.text;
        $("metric-tokens").textContent = String(event.tokens);
        $("result-status").textContent = `Generated ${event.tokens} token${event.tokens === 1 ? "" : "s"}…`;
        $("result-output").scrollTop = $("result-output").scrollHeight;
      } else if (event.type === "done") {
        showResult(event, "encode");
        $("covertext").value = event.text;
        completed = true;
      } else if (event.type === "error") {
        throw new Error(event.detail || "Encoding failed.");
      }
    }
    if (done) break;
  }
  if (!completed) throw new Error("The generation stream ended before completion.");
}

async function submit(form, mode) {
  $("error").hidden = true;
  $("result").hidden = true;
  setBusy(form, true, mode);
  const body = mode === "encode"
    ? {
        message: $("message").value,
        key: $("encode-key").value,
        ...codecOptions("encode"),
      }
    : {
        text: $("covertext").value,
        key: $("decode-key").value,
        ...codecOptions("decode"),
      };
  try {
    if (mode === "encode") {
      copyEncodeConfiguration();
      beginStream();
      await streamEncode(body);
    } else {
      showResult(await request("/api/decode", body), mode);
    }
  } catch (error) {
    $("error").textContent = error instanceof Error ? error.message : "Request failed.";
    $("error").hidden = false;
    $("result-status").textContent = "Generation stopped";
    $("result-status").classList.remove("live");
  } finally {
    setBusy(form, false, mode);
  }
}

$("encode-panel").addEventListener("submit", (event) => {
  event.preventDefault();
  submit(event.currentTarget, "encode");
});

$("decode-panel").addEventListener("submit", (event) => {
  event.preventDefault();
  submit(event.currentTarget, "decode");
});

$("copy").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(state.output);
    $("copy").textContent = "Copied";
    window.setTimeout(() => { $("copy").textContent = "Copy"; }, 1200);
  } catch {
    $("error").textContent = "Clipboard access was denied.";
    $("error").hidden = false;
  }
});

$("download").addEventListener("click", () => {
  const blob = new Blob([state.output], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = state.outputMode === "encode" ? "lec-covertext.txt" : "lec-message.txt";
  link.click();
  URL.revokeObjectURL(url);
});

configureEnvironment();
