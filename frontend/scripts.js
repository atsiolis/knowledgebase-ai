/**
 * Frontend JavaScript for KnowledgeBase AI
 * 
 * Handles:
 * - File upload with real-time progress tracking
 * - Document Q&A interactions
 * - Document list management and deletion
 * - UI updates and animations
 */

// ============================================================================
// CONFIGURATION
// ============================================================================

// Backend API URL - change this if running on different port
const API_URL = "http://localhost:8000";


// ============================================================================
// Q&A FUNCTIONALITY
// ============================================================================

/**
 * Handle user asking a question
 * 
 * Process:
 * 1. Get question from input field
 * 2. Display question in chat
 * 3. Call backend /ask endpoint
 * 4. Display answer with sources
 */
async function handleAsk() {
    const input = document.getElementById('userInput');
    const chatWindow = document.getElementById('chatWindow');
    const loading = document.getElementById('loading');
    const question = input.value.trim();

    // Don't send empty questions
    if (!question) return;

    // Remove welcome message if this is the first question
    const welcomeMsg = chatWindow.querySelector('.welcome-message');
    if (welcomeMsg) welcomeMsg.remove();

    // Step 1: Add user's question to chat
    chatWindow.innerHTML += `<div class="message user">${escapeHtml(question)}</div>`;
    input.value = "";  // Clear input field
    loading.classList.remove('hidden');  // Show "AI is thinking..." indicator
    chatWindow.scrollTop = chatWindow.scrollHeight;  // Auto-scroll to bottom

    try {
        // Step 2: Call backend API
        const response = await fetch(`${API_URL}/ask?question=${encodeURIComponent(question)}`);
        const data = await response.json();

        // Step 3: Prepare source citations
        // Remove duplicates and format as comma-separated list
        const sourceHtml = data.sources && data.sources.length > 0 
            ? `<div class="sources"><strong>Sources:</strong> ${[...new Set(data.sources)].join(", ")}</div>` 
            : "";

        // Step 4: Add bot's answer to chat
        chatWindow.innerHTML += `
            <div class="message bot">
                ${escapeHtml(data.answer)}
                ${sourceHtml}
            </div>
        `;
    } catch (error) {
        // Handle network errors
        chatWindow.innerHTML += `<div class="message bot" style="color: red;">Failed to connect to backend. Make sure the server is running.</div>`;
    } finally {
        // Always hide loading indicator and scroll to bottom
        loading.classList.add('hidden');
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }
}


// ============================================================================
// FILE UPLOAD FUNCTIONALITY
// ============================================================================

/**
 * Handle file upload
 * 
 * Process:
 * 1. Read selected file
 * 2. Send to backend
 * 3. Start polling for progress updates
 */
async function handleUpload() {
    const status = document.getElementById('uploadStatus');
    const fileInput = document.getElementById('fileInput');
    
    // Check if file was selected
    if (!fileInput.files[0]) return;

    // Prepare file for upload
    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    status.innerHTML = "⏳ Uploading...";
    
    try {
        // Send file to backend
        const resp = await fetch(`${API_URL}/upload`, { 
            method: "POST", 
            body: formData 
        });
        const res = await resp.json();
        
        if (res.status === "processing" && res.upload_id) {
            // File uploaded successfully - start polling for progress
            pollProgress(res.upload_id);
        } else if (res.status === "error") {
            // Upload failed
            status.innerHTML = `❌ Error: ${res.detail}`;
        }
    } catch (e) {
        // Network error
        status.innerHTML = "❌ Upload Failed";
    }
}

/**
 * Poll backend for upload progress updates
 * 
 * Checks progress every 500ms and updates the progress bar.
 * Stops when upload is complete or encounters an error.
 * 
 * @param {string} uploadId - Unique ID for this upload
 */
async function pollProgress(uploadId) {
    const status = document.getElementById('uploadStatus');
    
    // Poll every 500ms (twice per second)
    const pollInterval = setInterval(async () => {
        try {
            // Get current progress from backend
            const resp = await fetch(`${API_URL}/upload/status/${uploadId}`);
            const data = await resp.json();
            
            // Check if upload was not found
            if (data.status === "not_found") {
                clearInterval(pollInterval);
                status.innerHTML = "❌ Upload not found";
                return;
            }
            
            // Update progress bar and message
            const progress = data.progress || 0;
            status.innerHTML = `
                <div class="progress-container">
                    <div class="progress-bar" style="width: ${progress}%"></div>
                </div>
                <div class="progress-text">${data.message}</div>
            `;
            
            // Stop polling when complete or error
            if (data.status === "complete") {
                clearInterval(pollInterval);
                setTimeout(() => {
                    status.innerHTML = `<div class="progress-text" style="color: #059669;">${data.message}</div>`;
                    loadDocuments(); // Refresh document list
                }, 500);
            } else if (data.status === "error") {
                clearInterval(pollInterval);
                status.innerHTML = `<div class="progress-text" style="color: #dc2626;">${data.message}</div>`;
            }
        } catch (e) {
            // Network error during polling
            clearInterval(pollInterval);
            status.innerHTML = "❌ Connection error";
        }
    }, 500); // Poll every 500ms for smooth updates
}


// ============================================================================
// DOCUMENT MANAGEMENT
// ============================================================================

/**
 * Load and display all documents from the database
 * 
 * Fetches documents from backend and renders them in the sidebar.
 * Shows empty state if no documents exist.
 */
async function loadDocuments() {
    try {
        // Fetch documents from backend
        const resp = await fetch(`${API_URL}/documents`);
        const docs = await resp.json();
        
        const docsList = document.getElementById('documentsList');
        
        // Show empty state if no documents
        if (!docs || docs.length === 0) {
            docsList.innerHTML = `
                <div class="empty-state">
                    <i class="fa-solid fa-folder-open"></i>
                    <p>No documents yet.<br>Upload a file to get started.</p>
                </div>
            `;
            return;
        }
        
        // Render document list
        docsList.innerHTML = docs.map(doc => `
            <div class="document-item">
                <i class="fa-solid fa-file-pdf"></i>
                <span class="document-name">${escapeHtml(doc.name)}</span>
                <button class="delete-btn" data-doc-id="${doc.id}" data-doc-name="${escapeHtml(doc.name)}" title="Delete document">
                    <i class="fa-solid fa-trash"></i>
                </button>
            </div>
        `).join('');
        
        // Attach click handlers to delete buttons
        // Using event listeners instead of inline onclick to handle special characters
        document.querySelectorAll('.delete-btn').forEach(btn => {
            btn.addEventListener('click', function(e) {
                e.stopPropagation();  // Prevent event bubbling
                const docId = this.getAttribute('data-doc-id');
                const docName = this.getAttribute('data-doc-name');
                deleteDocument(docId, docName);
            });
        });
    } catch (e) {
        console.error('Failed to load documents:', e);
    }
}

/**
 * Delete a document and all its chunks
 * 
 * Shows confirmation dialog before deletion.
 * Refreshes document list on success.
 * 
 * @param {string} docId - UUID of document to delete
 * @param {string} docName - Name of document (for confirmation message)
 */
async function deleteDocument(docId, docName) {
    // Show confirmation dialog
    if (!confirm(`Delete "${docName}"?\n\nThis will remove the document and all associated data.`)) {
        return;  // User cancelled
    }
    
    try {
        // Call backend DELETE endpoint
        const resp = await fetch(`${API_URL}/documents/${docId}`, {
            method: 'DELETE'
        });
        const result = await resp.json();
        
        if (result.status === 'success') {
            // Refresh the document list
            loadDocuments();
            
            // Show success message briefly
            const status = document.getElementById('uploadStatus');
            status.innerHTML = '<div class="progress-text" style="color: #059669;">✅ Document deleted successfully</div>';
            setTimeout(() => {
                status.innerHTML = '';
            }, 3000);  // Clear after 3 seconds
        } else {
            // Deletion failed
            alert(`Failed to delete: ${result.message}`);
        }
    } catch (e) {
        // Network error
        alert('Failed to delete document. Please try again.');
        console.error('Delete error:', e);
    }
}


// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Escape HTML to prevent XSS attacks
 * 
 * Converts HTML special characters to their entity equivalents.
 * 
 * @param {string} text - Text to escape
 * @returns {string} - Escaped text safe for innerHTML
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}


// ============================================================================
// EVENT LISTENERS
// ============================================================================

// Add 'Enter' key support for sending messages
document.getElementById("userInput").addEventListener("keypress", (e) => {
    if (e.key === "Enter") handleAsk();
});

// Trigger upload when file is selected
document.getElementById("fileInput").onchange = handleUpload;

// Load documents when page loads
document.addEventListener('DOMContentLoaded', () => {
    loadDocuments();
});
