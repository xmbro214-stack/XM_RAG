const indexState = document.querySelector("#indexState");
const indexButton = document.querySelector("#indexButton");
const newChatButton = document.querySelector("#newChatButton");
const messages = document.querySelector("#messages");
const askForm = document.querySelector("#askForm");
const questionInput = document.querySelector("#questionInput");
const sendButton = document.querySelector("#sendButton");
const imageButton = document.querySelector("#imageButton");
const imageInput = document.querySelector("#imageInput");
const attachmentPreview = document.querySelector("#attachmentPreview");
const sourceCards = document.querySelector("#sourceCards");
const sourcePreview = document.querySelector("#sourcePreview");
const quickPrompts = document.querySelector(".quick-prompts");
const history = document.querySelector(".history");

let isAsking = false;
let selectedImages = [];

async function safeReadError(response, fallback) {
  try {
    const data = await response.clone().json();
    if (data && typeof data.detail === "string") {
      return data.detail;
    }
  } catch {
    // Fall through to plain-text response parsing.
  }

  try {
    const text = await response.text();
    return text.trim() || fallback;
  } catch {
    return fallback;
  }
}

function setIndexState(text, state) {
  indexState.textContent = text;
  indexState.className = `status-pill ${state}`;
}

async function refreshStatus() {
  try {
    const response = await fetch("/api/status");
    if (!response.ok) {
      throw new Error(await safeReadError(response, "检查知识库失败"));
    }
    const status = await response.json();
    if (status.indexed) {
      setIndexState("索引已就绪", "ready");
      return;
    }
    setIndexState("索引未构建", "missing");
  } catch (error) {
    setIndexState("知识库异常", "missing");
  }
}

function formatScore(score) {
  const value = Number(score);
  return Number.isFinite(value) ? `${Math.round(value * 100)}%` : "未知";
}

function getConfidence(sources) {
  const topScore = Number(sources?.[0]?.score ?? 0);
  if (topScore >= 0.68) return "高";
  if (topScore >= 0.5) return "中";
  return "低";
}

function createSection(title, icon, contentNode) {
  const section = document.createElement("section");
  section.className = "answer-section";

  const heading = document.createElement("h3");
  heading.className = "section-title";

  const iconNode = document.createElement("span");
  iconNode.className = "section-icon";
  iconNode.textContent = icon;

  const titleNode = document.createElement("span");
  titleNode.textContent = title;

  heading.append(iconNode, titleNode);
  section.append(heading, contentNode);
  return section;
}

function renderSourceChips(sources) {
  const list = document.createElement("div");
  list.className = "source-chips";

  for (const source of sources) {
    const chip = document.createElement("span");
    chip.className = "source-chip";
    chip.textContent = `${source.doc_title} 第 ${source.page} 页`;
    list.appendChild(chip);
  }

  return list;
}

function renderSourcesPanel(sources) {
  sourceCards.replaceChildren();

  if (!sources || sources.length === 0) {
    const defaultDocs = [
      ["问界 M6 说明书 v2.1", "PDF · 388 页 · 本地知识库", "已索引"],
      ["仪表指示灯与故障灯", "重点结构化 · 95 条图标", "高相关"],
    ];
    for (const [titleText, metaText, scoreText] of defaultDocs) {
      sourceCards.appendChild(createSourceCard(titleText, metaText, scoreText));
    }
    renderSourcePreview({
      doc_title: "问界 M6 说明书 v2.1",
      page: "302",
      snippet: "可预览答案引用片段、页码和相关章节。提问后这里会展示最相关来源。",
    });
    return;
  }

  for (const source of sources) {
    sourceCards.appendChild(
      createSourceCard(source.doc_title, `第 ${source.page} 页`, `相似度 ${formatScore(source.score)}`)
    );
  }

  renderSourcePreview(sources[0]);
}

function createSourceCard(titleText, metaText, scoreText) {
  const card = document.createElement("article");
  card.className = "source-card";

  const icon = document.createElement("span");
  icon.className = "doc-icon";
  icon.textContent = "▣";

  const body = document.createElement("div");
  const title = document.createElement("strong");
  title.textContent = titleText;
  const meta = document.createElement("p");
  meta.textContent = metaText;
  body.append(title, meta);

  const score = document.createElement("span");
  score.className = "source-score";
  score.textContent = scoreText;

  card.append(icon, body, score);
  return card;
}

function renderSourcePreview(source) {
  sourcePreview.replaceChildren();

  const vehicle = document.createElement("div");
  vehicle.className = "preview-vehicle";
  vehicle.setAttribute("aria-hidden", "true");
  vehicle.appendChild(document.createElement("span"));

  const title = document.createElement("strong");
  title.textContent = source.doc_title;
  const meta = document.createElement("span");
  meta.textContent = `第 ${source.page} 页`;
  const snippet = document.createElement("p");
  snippet.textContent = source.snippet;

  sourcePreview.append(vehicle, title, meta, snippet);
}

function renderAttachmentPreview() {
  attachmentPreview.replaceChildren();

  if (selectedImages.length === 0) {
    attachmentPreview.hidden = true;
    return;
  }

  attachmentPreview.hidden = false;
  for (const file of selectedImages) {
    const item = document.createElement("div");
    item.className = "attachment-chip";

    const thumbnail = document.createElement("img");
    thumbnail.alt = "";
    thumbnail.src = URL.createObjectURL(file);
    thumbnail.onload = () => URL.revokeObjectURL(thumbnail.src);

    const name = document.createElement("span");
    name.textContent = file.name;

    item.append(thumbnail, name);
    attachmentPreview.appendChild(item);
  }
}

function addSelectedImages(files) {
  const images = files.filter((file) => file.type.startsWith("image/"));
  if (images.length === 0) return false;
  selectedImages = [...selectedImages, ...images];
  renderAttachmentPreview();
  return true;
}

function clearImages() {
  selectedImages = [];
  imageInput.value = "";
  renderAttachmentPreview();
}

function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

function addUserMessage(text, images = []) {
  const node = document.createElement("article");
  node.className = "message user";

  const content = document.createElement("div");
  content.textContent = text;
  node.appendChild(content);

  if (images.length > 0) {
    const list = document.createElement("div");
    list.className = "message-attachments";
    for (const file of images) {
      const image = document.createElement("img");
      image.alt = file.name;
      image.src = URL.createObjectURL(file);
      image.onload = () => URL.revokeObjectURL(image.src);
      list.appendChild(image);
    }
    node.appendChild(list);
  }

  messages.appendChild(node);
  messages.scrollTop = messages.scrollHeight;
}

function addLoadingMessage() {
  const node = document.createElement("article");
  node.className = "message assistant-card loading-card";

  const confidence = document.createElement("div");
  confidence.className = "confidence";
  confidence.textContent = "正在生成回答";

  const title = document.createElement("h3");
  title.className = "section-title";
  title.textContent = "检索知识库并组织答案...";

  const progress = document.createElement("div");
  progress.className = "progress-bar";
  const indicator = document.createElement("span");
  progress.appendChild(indicator);

  const hint = document.createElement("p");
  hint.className = "loading-hint";
  hint.textContent = "请稍候，模型返回前会一直显示此进度。";

  node.append(confidence, title, progress, hint);
  messages.appendChild(node);
  messages.scrollTop = messages.scrollHeight;
  return node;
}

function addAssistantMessage(answer, sources = []) {
  const node = document.createElement("article");
  node.className = "message assistant-card";

  const confidence = document.createElement("div");
  confidence.className = "confidence";
  confidence.textContent = `回答可信度：${getConfidence(sources)}`;
  node.appendChild(confidence);

  const conclusion = document.createElement("p");
  conclusion.className = "answer-text";
  conclusion.textContent = answer;
  node.appendChild(createSection("结论与建议", "✓", conclusion));

  if (sources.length > 0) {
    node.appendChild(createSection("引用来源", "文", renderSourceChips(sources)));
  }

  messages.appendChild(node);
  messages.scrollTop = messages.scrollHeight;
  renderSourcesPanel(sources);
}

function addSystemMessage(text) {
  addAssistantMessage(text, []);
}

function vehicleLogoMarkup() {
  return `
    <figure class="m6-photo-card">
      <img class="m6-photo" src="/static/assets/wenjie-m6-photo.svg" alt="AITO Wenjie M6">
    </figure>
  `;
}

function addWelcomePanel() {
  const panel = document.createElement("section");
  panel.className = "welcome-panel";

  const copyWrap = document.createElement("div");
  copyWrap.className = "welcome-copy";

  const icon = document.createElement("div");
  icon.className = "welcome-icon";
  icon.textContent = "M6";

  const copy = document.createElement("div");
  const title = document.createElement("h2");
  title.textContent = "您好，有什么可以帮助您？";
  const subtitle = document.createElement("p");
  subtitle.textContent = "基于「问界 M6 说明书」知识库，为您提供用车、充电、故障和安全提示查询。";
  copy.append(title, subtitle);
  copyWrap.append(icon, copy);

  const concept = document.createElement("div");
  concept.className = "m6-concept";
  concept.setAttribute("aria-hidden", "true");
  concept.innerHTML = vehicleLogoMarkup();

  panel.append(copyWrap, concept);
  messages.appendChild(panel);
}

function clearConversation() {
  messages.replaceChildren();
  addWelcomePanel();
  renderSourcesPanel([]);
  clearImages();
  setActiveHistoryItem(null);
  questionInput.focus();
}

function setActiveHistoryItem(activeItem) {
  document.querySelectorAll(".history-item").forEach((item) => {
    item.classList.toggle("active", item === activeItem);
  });
}

function submitQuestion(text) {
  questionInput.value = text;
  askForm.requestSubmit();
}

indexButton.addEventListener("click", async () => {
  indexButton.disabled = true;
  indexButton.textContent = "更新中...";
  try {
    const response = await fetch("/api/index", { method: "POST" });
    if (!response.ok) {
      throw new Error(await safeReadError(response, "构建索引失败"));
    }
    const result = await response.json();
    const message = result.reused
      ? `知识库已就绪，共 ${result.chunks} 个片段。`
      : `知识库已更新，共 ${result.chunks} 个片段。`;
    addSystemMessage(message);
    await refreshStatus();
  } catch (error) {
    addSystemMessage(error.message);
  } finally {
    indexButton.disabled = false;
    indexButton.textContent = "管理知识库";
  }
});

newChatButton.addEventListener("click", clearConversation);

imageButton.addEventListener("click", () => {
  imageInput.click();
});

imageInput.addEventListener("change", () => {
  addSelectedImages(Array.from(imageInput.files || []));
  imageInput.value = "";
});

document.addEventListener("paste", (event) => {
  const files = Array.from(event.clipboardData?.files || []);
  if (addSelectedImages(files)) {
    event.preventDefault();
    questionInput.focus();
  }
});

quickPrompts.addEventListener("click", (event) => {
  const button = event.target.closest("button");
  if (!button) return;
  questionInput.value = button.textContent.trim();
  questionInput.focus();
});

history.addEventListener("click", (event) => {
  const item = event.target.closest(".history-item");
  if (!item) return;
  const question = item.dataset.question || item.querySelector("strong")?.textContent;
  if (!question || isAsking) return;
  setActiveHistoryItem(item);
  submitQuestion(question.trim());
});

askForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (isAsking) return;

  const question = questionInput.value.trim();
  if (!question) return;

  isAsking = true;
  const imagesForMessage = [...selectedImages];
  addUserMessage(question, imagesForMessage);
  const loadingMessage = addLoadingMessage();
  questionInput.value = "";
  clearImages();
  questionInput.disabled = true;
  imageButton.disabled = true;
  sendButton.disabled = true;
  sendButton.textContent = "发送中";

  try {
    const imageDataUrls = await Promise.all(imagesForMessage.map(fileToDataUrl));
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, images: imageDataUrls }),
    });
    if (!response.ok) {
      throw new Error(await safeReadError(response, "请求失败"));
    }
    const result = await response.json();
    loadingMessage.remove();
    addAssistantMessage(result.answer, result.sources || []);
  } catch (error) {
    loadingMessage.remove();
    addSystemMessage(error.message);
  } finally {
    isAsking = false;
    questionInput.disabled = false;
    imageButton.disabled = false;
    sendButton.disabled = false;
    sendButton.textContent = "发送";
    questionInput.focus();
  }
});

refreshStatus();
