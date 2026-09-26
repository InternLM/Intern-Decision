"use strict";
const $ = (id) => document.getElementById(id);
let presets = [], records = [], uploads = [], busy = false, ready = false, controller = null;
let uploadQueue = Promise.resolve(), readingImages = 0;
const imageTypes = new Set(["image/jpeg", "image/png", "image/webp", "image/gif"]);
function syncControls() {
  $("run").disabled = busy || !ready || readingImages > 0;
  $("choose-images").disabled = busy || readingImages > 0;
  $("examples").disabled = busy;
  $("input-mode").textContent = uploads.length ? "图文模式 · " + uploads.length + " 张图片" : "文本 / 图文";
  $("image-hint").textContent = readingImages ? "正在读取图片…" : "JPEG / PNG / WebP / 静态 GIF · 最多 8 张 · 总计 32 MB";
}
const fmt = (value) => Number(value).toLocaleString("en-US", {maximumFractionDigits: 1});
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
function chooseExample() {
  const item = presets[Number($("examples").value)];
  $("state").value = typeof item.state === "string" ? item.state : JSON.stringify(item.state, null, 2);
  $("questions").value = JSON.stringify(item.questions, null, 2);
  $("error").textContent = "";
  uploads = []; renderUploads();
}
function renderUploads() {
  const preview = $("image-preview");
  preview.replaceChildren();
  uploads.forEach((file, index) => {
    const item = el("div", "image-thumb");
    const image = document.createElement("img");
    image.src = file.data;
    image.alt = file.name;
    const remove = el("button", "", "×");
    remove.type = "button";
    remove.disabled = busy;
    remove.setAttribute("aria-label", "移除 " + file.name);
    remove.title = "移除 " + file.name;
    remove.addEventListener("click", () => { if (busy) return; uploads.splice(index, 1); renderUploads(); });
    item.append(image, remove);
    preview.append(item);
  });
  syncControls();
}
function addFiles(fileList) {
  if (busy) return;
  const files = Array.from(fileList || []);
  if (!files.length) return;
  readingImages++;
  syncControls();
  uploadQueue = uploadQueue.then(async () => {
    if (uploads.length + files.length > 8) throw Error("最多上传 8 张图片。");
    if (uploads.reduce((sum, file) => sum + file.size, 0) + files.reduce((sum, file) => sum + file.size, 0) > 32 * 1024 * 1024) {
      throw Error("图片总计不能超过 32 MB。");
    }
    for (const file of files) {
      if (!imageTypes.has(file.type)) throw Error("只支持 JPEG、PNG、WebP 和静态 GIF。");
      if (!file.size || file.size > 12 * 1024 * 1024) throw Error("单张图片应小于 12 MB，且不能为空。");
    }
    const items = await Promise.all(files.map(file => new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve({name: file.name, type: file.type, size: file.size, data: reader.result});
      reader.onerror = () => reject(Error("读取图片失败。"));
      reader.readAsDataURL(file);
    })));
    uploads.push(...items);
    $("error").textContent = "";
    renderUploads();
  }).catch(error => { $("error").textContent = "图片上传失败：" + error.message; })
    .finally(() => { readingImages--; syncControls(); });
}
function distribution(parent, answer) {
  for (const [option, probability] of Object.entries(answer.probabilities || {})) {
    const label = el("div", "prob-label");
    label.append(el("span", "", option), el("span", "", (probability * 100).toFixed(2) + "%"));
    const track = el("div", "track"), bar = el("div", "bar" + (option === answer.decision ? " best" : ""));
    bar.style.width = Math.max(0, Math.min(100, probability * 100)) + "%";
    track.append(bar); parent.append(label, track);
  }
}
function renderCard(card, field, answer, state) {
  const heading = el("div", "answer-head");
  heading.append(el("span", "field-name", field), el("span", "type", answer.type));
  let title = answer.decision;
  if (answer.type === "noul") title = answer.decision === "yes" ? "是 / Yes" : "否 / No";
  else if (answer.type === "score") title = fmt(answer.score);
  card.replaceChildren(heading, el("div", "selected", title));
  if (answer.source === "thinking") {
    card.append(el("p", "help", "深度思考 · 最终选择（不提供概率估计）"));
    const details = el("details", "details");
    details.append(el("summary", "", "查看初始预测"));
    distribution(details, answer.initial);
    card.append(details);
  } else {
    card.append(el("p", "help", "初始预测 · 置信度 " + (answer.confidence * 100).toFixed(1) + "%"));
    distribution(card, answer);
  }
  if (state) {
    const status = el("p", state.error ? "error" : "status", state.label);
    const rate = state.startedAt && state.tokenCount
      ? Math.round(state.tokenCount / Math.max(0.001, (state.lastAt - state.startedAt) / 1000))
      : 0;
    const metric = el("div", "stream-metric", rate ? "约 " + rate + " tokens/s" : "等待首个 token");
    const stream = document.createElement("textarea");
    stream.className = "stream-output";
    stream.rows = 1;
    stream.readOnly = true;
    stream.setAttribute("aria-label", "思考模型实时输出");
    stream.value = state.text || "等待流式输出…";
    card.append(metric, stream, status);
  }
}
function renderHistory() {
  $("history").replaceChildren();
  for (const item of records.slice().reverse()) {
    const row = el("tr");
    const state = typeof item.request.state === "string" ? item.request.state : JSON.stringify(item.request.state);
    const values = [new Date(item.timestamp).toLocaleTimeString(), state.slice(0, 36),
      item.response.usage.decision_count, fmt(item.response.timing.inference_ms) + " ms",
      fmt(item.response.timing.server_ms) + " ms", fmt(item.roundtrip_ms) + " ms"];
    for (const value of values) row.append(el("td", "", String(value)));
    $("history").append(row);
  }
  $("download").disabled = records.length === 0;
}
async function consumeSSE(response, onEvent) {
  if (!response.ok) {
    const data = await response.json();
    throw Error(typeof data.detail === "string" ? data.detail : "思考请求失败");
  }
  if (!response.body) throw Error("浏览器不支持流式响应");
  const reader = response.body.getReader(), decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const {value, done} = await reader.read();
      buffer += decoder.decode(value, {stream: !done});
      // Parse lines so CRLF split across network chunks remains valid.
      let end;
      while ((end = buffer.indexOf("\n\n")) !== -1 || (end = buffer.indexOf("\r\n\r\n")) !== -1) {
        const crlf = buffer.slice(end, end + 4) === "\r\n\r\n";
        const block = buffer.slice(0, end);
        buffer = buffer.slice(end + (crlf ? 4 : 2));
        const data = block.split(/\r?\n/).filter(line => line.startsWith("data:"))
          .map(line => line.slice(5).trimStart()).join("\n");
        if (data) onEvent(JSON.parse(data));
      }
      if (done) break;
    }
  } finally { reader.releaseLock(); }
}
async function run() {
  if (busy || !ready || readingImages) return;
  $("error").textContent = "";
  let request;
  try {
    const questions = JSON.parse($("questions").value);
    if (!questions || typeof questions !== "object" || Array.isArray(questions)) throw Error("Questions 必须为 JSON 对象。");
    if (!$("state").value.trim() && !uploads.length) throw Error("请输入场景或上传图片。");
    const threshold = Number($("threshold").value);
    if (!$("threshold").value || !Number.isFinite(threshold) || threshold < 0 || threshold > 1) throw Error("置信度阈值应在 0 与 1 之间。");
    request = {state: $("state").value, questions,
      thinking: {enabled: $("thinking").checked, confidence_threshold: threshold}};
    if (uploads.length) request.images = uploads.map(file => ({name: file.name, type: file.type, data: file.data}));
  } catch (error) { $("error").textContent = "输入有误：" + error.message; return; }
  busy = true;
  syncControls();
  renderUploads();
  controller = new AbortController();
  const signal = controller.signal;
  $("run").disabled = true;
  $("cancel").hidden = false;
  $("run").textContent = "正在推理…";
  $("status").textContent = "正在生成初始决策…";
  $("result-panel").setAttribute("aria-busy", "true");
  const start = performance.now();
  let record;
  try {
    const response = await fetch("/v1/decisions", {method: "POST", signal,
      headers: {"Content-Type": "application/json"}, body: JSON.stringify(request)});
    const data = await response.json();
    if (!response.ok) throw Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail || data));
    const roundtrip = performance.now() - start;
    const cards = new Map(), states = new Map();
    $("answers").replaceChildren();
    for (const [field, answer] of Object.entries(data.answers)) {
      const card = el("div", "answer"); cards.set(field, card); $("answers").append(card);
      renderCard(card, field, answer);
    }
    $("forward").textContent = fmt(data.timing.inference_ms);
    $("server").textContent = fmt(data.timing.server_ms);
    $("roundtrip").textContent = fmt(roundtrip);
    $("usage").textContent = data.usage.input_tokens + " tokens · " + data.usage.decision_count + " 个字段";
    const recordRequest = {...request};
    if (request.images) recordRequest.images = request.images.map(({name, type}) => ({name, type}));
    record = {timestamp: new Date().toISOString(), request: recordRequest, response: data, roundtrip_ms: roundtrip};
    records.push(record);
    const refresh = () => {
      $("raw").textContent = JSON.stringify(data, null, 2);
      renderHistory();
    };
    refresh(); // Initial JSON is visible before any external stream starts.
    const tasks = data.thinking.tasks;
    $("status").textContent = tasks.length ? "初始决策已返回 · " + tasks.length + " 个问题正在并行思考" : "决策完成";
    let failures = 0;
    const outcomes = await Promise.allSettled(tasks.map(async task => {
      const state = {label: "正在思考…", text: "", error: false,
        tokenCount: 0, startedAt: performance.now(), lastAt: performance.now()};
      states.set(task.field, state);
      const draw = () => renderCard(cards.get(task.field), task.field, data.answers[task.field], state);
      draw();
      let terminal = false;
      try {
        const stream = await fetch(task.stream_url, {method: "POST", signal,
          headers: {"Content-Type": "application/json"}, body: JSON.stringify({field: task.field, token: task.token})});
        await consumeSSE(stream, event => {
          if (event.field !== task.field) throw Error("响应字段不匹配");
          if (event.event === "delta") {
            state.text += event.text;
            state.tokenCount += Math.max(1, Math.ceil(Array.from(event.text).length / 4));
            state.lastAt = performance.now();
            draw();
          }
          else if (event.event === "final") {
            data.answers[task.field] = event.answer;
            state.label = "思考完成 · 最终决策已更新"; terminal = true;
            data.thinking_timings = data.thinking_timings || {};
            data.thinking_timings[task.field] = event.thinking_ms;
            draw(); refresh();
          } else if (event.event === "error") { throw Error(event.message); }
        });
        if (!terminal) throw Error("思考响应中断，保留初始决策");
      } catch (error) {
        failures++;
        state.error = true;
        state.label = signal.aborted ? "已取消思考 · 保留初始决策" : "思考失败 · 保留初始决策：" + error.message;
        data.thinking_errors = data.thinking_errors || {};
        data.thinking_errors[task.field] = state.label;
        draw(); refresh();
      }
    }));
    failures += outcomes.filter(item => item.status === "rejected").length;
    record.roundtrip_ms = performance.now() - start;
    $("roundtrip").textContent = fmt(record.roundtrip_ms);
    $("status").textContent = signal.aborted ? "已取消 · 保留已返回的结果" : failures ? "已完成 · 部分问题保留初始决策" : "决策完成";
    refresh();
  } catch (error) {
    $("error").textContent = signal.aborted ? "请求已取消" : "请求失败：" + error.message;
    $("status").textContent = "请求未完成";
  } finally {
    busy = false; controller = null;
    syncControls(); renderUploads(); $("run").textContent = "运行决策 →";
    $("cancel").hidden = true;
    $("result-panel").setAttribute("aria-busy", "false");
  }
}
$("run").addEventListener("click", run);
$("choose-images").addEventListener("click", () => $("image-files").click());
$("image-files").addEventListener("change", event => { addFiles(event.target.files); event.target.value = ""; });
$("dropzone").addEventListener("dragover", event => { event.preventDefault(); $("dropzone").classList.add("dragover"); });
$("dropzone").addEventListener("dragleave", () => $("dropzone").classList.remove("dragover"));
$("dropzone").addEventListener("drop", event => { event.preventDefault(); $("dropzone").classList.remove("dragover"); addFiles(event.dataTransfer.files); });
$("dropzone").addEventListener("keydown", event => {
  if (event.target === $("dropzone") && (event.key === "Enter" || event.key === " ")) {
    event.preventDefault(); if (!busy && !readingImages) $("image-files").click();
  }
});
$("cancel").addEventListener("click", () => controller?.abort());
window.addEventListener("pagehide", () => controller?.abort());
$("examples").addEventListener("change", chooseExample);
document.addEventListener("keydown", event => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") { event.preventDefault(); run(); }
});
$("download").addEventListener("click", () => {
  const url = URL.createObjectURL(new Blob([JSON.stringify(records, null, 2)], {type: "application/json"}));
  const link = el("a"); link.href = url; link.download = "interndecision-" + Date.now() + ".json";
  document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
});
async function init() {
  try {
    const responses = await Promise.all([fetch("/examples"), fetch("/health"), fetch("/demo-meta")]);
    if (responses.some(response => !response.ok)) throw Error("服务连接失败，请刷新重试。");
    const [examples, health, meta] = await Promise.all(responses.map(response => response.json()));
    presets = examples;
    $("examples").replaceChildren();
    presets.forEach((example, i) => { const option = el("option", "", example.title); option.value = String(i); $("examples").append(option); });
    $("examples").disabled = false; chooseExample();
    $("thinking").disabled = !meta.thinking.available;
    $("thinking").checked = meta.thinking.available && meta.thinking.enabled;
    $("threshold").value = meta.thinking.confidence_threshold;
    $("thinking-help").textContent = meta.thinking.available
      ? "低于阈值的问题将并行交给思考模型，流式返回最终选择。"
      : "思考服务尚未配置；当前可使用初始决策。";
    ready = health.model_loaded;
    $("health").textContent = ready ? "已就绪" : "尚未就绪";
    $("health-dot").classList.toggle("ready", ready); syncControls();
  } catch (error) { $("health").textContent = "连接失败"; $("error").textContent = error.message; }
}
init();
