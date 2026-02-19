/**
 * Frontend JavaScript for KnowledgeBase AI
 *
 * Handles:
 * - File upload with real-time progress tracking
 * - Streaming Q&A via Server-Sent Events (SSE)
 * - Document list management and deletion
 * - UI updates and animations
 */

// ============================================================================
// CONFIGURATION
// ============================================================================

const API_URL = "http://localhost:8000";


// ============================================================================
// Q&A — STREAMING
// ============================================================================

/**
 * Handle user asking a question.
 *
 * Connects to /ask via fetch() and reads the SSE stream manually using
 * the ReadableStream API. Tokens are appended to the message bubble as
 * they arrive, re-rendering markdown on every chunk so the user sees
 * the answer grow in real time — just like ChatGPT.
 *
 * SSE event types:
 *   { type: "sources", content: ["file.pdf", ...] }
 *   { type: "token",   content: "Hello" }
 *   { type: "done" }
 *   { type: "error",   content: "..." }
 */
async function handleAsk() {
    const input      = document.getElementById("userInput");
    const chatWindow = document.getElementById("chatWindow");
    const loading    = document.getElementById("loading");
    const sendBtn    = document.getElementById("sendBtn");
    const question   = input.value.trim();

    if (!question) return;

    // Remove welcome message on first question
    const welcome = chatWindow.querySelector(".welcome-message");
    if (welcome) welcome.remove();

    // Show user bubble
    chatWindow.innerHTML += `<div class="message user">${escapeHtml(question)}</div>`;
    input.value = "";

    // Disable input while streaming
    input.disabled = true;
    sendBtn.disabled = true;
    loading.classList.remove("hidden");
    chatWindow.scrollTop = chatWindow.scrollHeight;

    // Create the bot bubble that we'll stream into
    const botMsgId = `bot-${Date.now()}`;
    chatWindow.innerHTML += `<div class="message bot streaming" id="${botMsgId}"></div>`;
    const botBubble = document.getElementById(botMsgId);

    let accumulatedText = "";
    let sources = [];

    try {
        const response = await fetch(
            `${API_URL}/ask?question=${encodeURIComponent(question)}`
        );

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        // ReadableStream reader — processes SSE chunks as they arrive
        const reader  = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer    = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            // Decode the chunk and append to our line buffer
            buffer += decoder.decode(value, { stream: true });

            // SSE events are separated by double newlines
            const events = buffer.split("\n\n");
            // Keep the last (potentially incomplete) event in the buffer
            buffer = events.pop();

            for (const event of events) {
                const line = event.trim();
                if (!line.startsWith("data: ")) continue;

                let data;
                try {
                    data = JSON.parse(line.slice(6));
                } catch {
                    continue;
                }

                if (data.type === "sources") {
                    // Store sources — we'll render them after the answer is done
                    sources = data.content;

                } else if (data.type === "token") {
                    // Append token and re-render markdown in the bubble
                    accumulatedText += data.content;
                    botBubble.innerHTML = marked.parse(accumulatedText);
                    chatWindow.scrollTop = chatWindow.scrollHeight;

                } else if (data.type === "done") {
                    // Append source citations below the answer
                    if (sources.length > 0) {
                        const sourceHtml = `
                            <div class="sources">
                                <strong>Sources:</strong> ${[...new Set(sources)].join(", ")}
                            </div>`;
                        botBubble.innerHTML = marked.parse(accumulatedText) + sourceHtml;
                    }

                } else if (data.type === "error") {
                    botBubble.innerHTML = `<span style="color:#dc2626">${escapeHtml(data.content)}</span>`;
                }
            }
        }

    } catch (error) {
        botBubble.innerHTML =
            `<span style="color:#dc2626">Failed to connect to backend. Make sure the server is running.</span>`;
    } finally {
        // Remove streaming cursor class, re-enable UI
        botBubble.classList.remove("streaming");
        loading.classList.add("hidden");
        input.disabled = false;
        sendBtn.disabled = false;
        input.focus();
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }
}


// ============================================================================
// FILE UPLOAD
// ============================================================================

/**
 * Handle file upload.
 *
 * Sends the file to /upload, which returns immediately with an upload_id.
 * We then start polling /upload/status/{upload_id} every 500ms for progress.
 */
async function handleUpload() {
    const status    = document.getElementById("uploadStatus");
    const fileInput = document.getElementById("fileInput");

    if (!fileInput.files[0]) return;

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    status.innerHTML = "⏳ Uploading...";

    try {
        const resp = await fetch(`${API_URL}/upload`, { method: "POST", body: formData });
        const res  = await resp.json();

        if (res.status === "processing" && res.upload_id) {
            pollProgress(res.upload_id);
        } else if (res.status === "error") {
            status.innerHTML = `❌ Error: ${res.detail}`;
        }
    } catch {
        status.innerHTML = "❌ Upload failed";
    }
}

/**
 * Poll backend for upload progress every 500ms.
 * Stops when status is 'complete' or 'error'.
 *
 * @param {string} uploadId
 */
async function pollProgress(uploadId) {
    const status = document.getElementById("uploadStatus");

    const pollInterval = setInterval(async () => {
        try {
            const resp = await fetch(`${API_URL}/upload/status/${uploadId}`);
            const data = await resp.json();

            if (data.status === "not_found") {
                clearInterval(pollInterval);
                status.innerHTML = "❌ Upload not found";
                return;
            }

            const progress = data.progress || 0;
            status.innerHTML = `
                <div class="progress-container">
                    <div class="progress-bar" style="width:${progress}%"></div>
                </div>
                <div class="progress-text">${data.message}</div>`;

            if (data.status === "complete") {
                clearInterval(pollInterval);
                setTimeout(() => {
                    status.innerHTML = `<div class="progress-text" style="color:#059669">${data.message}</div>`;
                    loadDocuments();
                }, 500);
            } else if (data.status === "error") {
                clearInterval(pollInterval);
                status.innerHTML = `<div class="progress-text" style="color:#dc2626">${data.message}</div>`;
            }
        } catch {
            clearInterval(pollInterval);
            status.innerHTML = "❌ Connection error";
        }
    }, 500);
}


// ============================================================================
// DOCUMENT MANAGEMENT
// ============================================================================

async function loadDocuments() {
    try {
        const resp = await fetch(`${API_URL}/documents`);
        const docs = await resp.json();
        const docsList = document.getElementById("documentsList");

        if (!docs || docs.length === 0) {
            docsList.innerHTML = `
                <div class="empty-state">
                    <i class="fa-solid fa-folder-open"></i>
                    <p>No documents yet.<br>Upload a file to get started.</p>
                </div>`;
            return;
        }

        docsList.innerHTML = docs.map(doc => `
            <div class="document-item">
                <i class="fa-solid fa-file-pdf"></i>
                <span class="document-name">${escapeHtml(doc.name)}</span>
                <button class="delete-btn"
                        data-doc-id="${doc.id}"
                        data-doc-name="${escapeHtml(doc.name)}"
                        title="Delete document">
                    <i class="fa-solid fa-trash"></i>
                </button>
            </div>`).join("");

        document.querySelectorAll(".delete-btn").forEach(btn => {
            btn.addEventListener("click", function (e) {
                e.stopPropagation();
                deleteDocument(
                    this.getAttribute("data-doc-id"),
                    this.getAttribute("data-doc-name")
                );
            });
        });
    } catch (e) {
        console.error("Failed to load documents:", e);
    }
}

async function deleteDocument(docId, docName) {
    if (!confirm(`Delete "${docName}"?\n\nThis will remove the document and all associated data.`)) return;

    try {
        const resp   = await fetch(`${API_URL}/documents/${docId}`, { method: "DELETE" });
        const result = await resp.json();

        if (result.status === "success") {
            loadDocuments();
            const status = document.getElementById("uploadStatus");
            status.innerHTML = `<div class="progress-text" style="color:#059669">✅ Document deleted successfully</div>`;
            setTimeout(() => { status.innerHTML = ""; }, 3000);
        } else {
            alert(`Failed to delete: ${result.message}`);
        }
    } catch {
        alert("Failed to delete document. Please try again.");
    }
}


// ============================================================================
// UTILITIES
// ============================================================================

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}


// ============================================================================
// EVENT LISTENERS
// ============================================================================

document.getElementById("userInput").addEventListener("keypress", (e) => {
    if (e.key === "Enter" && !e.shiftKey) handleAsk();
});

document.getElementById("fileInput").onchange = handleUpload;

document.addEventListener("DOMContentLoaded", () => {
    loadDocuments();
});