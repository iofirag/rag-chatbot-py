import { useEffect, useMemo, useState } from 'react';

const API_BASE = '/api';

function App() {
  const [conversationId, setConversationId] = useState('default');
  const [messages, setMessages] = useState([]);
  const [messageInput, setMessageInput] = useState('');
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [availableFiles, setAvailableFiles] = useState([]);
  const [activeFileFilters, setActiveFileFilters] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [sending, setSending] = useState(false);
  const [status, setStatus] = useState('Ready');

  const canUpload = useMemo(() => selectedFiles.length > 0 && !uploading, [selectedFiles, uploading]);
  const canSend = useMemo(() => messageInput.trim().length > 0 && !sending, [messageInput, sending]);

  const loadFiles = async (id) => {
    try {
      const response = await fetch(`${API_BASE}/files/${encodeURIComponent(id)}`);
      if (!response.ok) throw new Error('Failed to load file metadata');
      const data = await response.json();
      const files = data.files || [];
      setAvailableFiles(files);
      setActiveFileFilters((prev) => prev.filter((item) => files.includes(item)));
    } catch (error) {
      setStatus(`File metadata error: ${error.message}`);
    }
  };

  const loadHistory = async (id) => {
    try {
      const response = await fetch(`${API_BASE}/history/${encodeURIComponent(id)}`);
      if (!response.ok) throw new Error('Failed to load history');
      const data = await response.json();
      setMessages(data.messages || []);
      setStatus(`Loaded ${data.messages?.length || 0} messages`);
      await loadFiles(id);
    } catch (error) {
      setStatus(`History error: ${error.message}`);
    }
  };

  useEffect(() => {
    loadHistory(conversationId);
  }, []);

  const onChangeConversation = async (event) => {
    event.preventDefault();
    await loadHistory(conversationId);
  };

  const onUpload = async () => {
    if (!canUpload) return;

    setUploading(true);
    setStatus('Uploading files and indexing chunks...');

    try {
      const form = new FormData();
      selectedFiles.forEach((file) => form.append('files', file));
      form.append('conversation_id', conversationId);

      const response = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: form,
      });
      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'Upload failed');
      }

      const data = await response.json();
      setSelectedFiles([]);
      setStatus(`Indexed ${data.total_chunks} chunks from ${data.files.length} files`);
      await loadFiles(conversationId);
    } catch (error) {
      setStatus(`Upload error: ${error.message}`);
    } finally {
      setUploading(false);
    }
  };

  const toggleFilter = (filename) => {
    setActiveFileFilters((prev) =>
      prev.includes(filename) ? prev.filter((item) => item !== filename) : [...prev, filename],
    );
  };

  const parseNdjsonChunks = (buffer) => {
    const lines = buffer.split('\n');
    const remainder = lines.pop() || '';
    const events = [];
    for (const line of lines) {
      if (!line.trim()) continue;
      events.push(JSON.parse(line));
    }
    return { events, remainder };
  };

  const onSend = async (event) => {
    event.preventDefault();
    if (!canSend) return;

    const text = messageInput.trim();
    setMessageInput('');
    setSending(true);

    setMessages((prev) => [...prev, { role: 'user', text, created_at: new Date().toISOString() }]);
    const assistantIndex = messages.length + 1;
    setMessages((prev) => [...prev, { role: 'assistant', text: '', contexts: [], streaming: true, created_at: new Date().toISOString() }]);
    setStatus('Generating answer with retrieved context...');

    try {
      const response = await fetch(`${API_BASE}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          conversation_id: conversationId,
          message: text,
          file_filters: activeFileFilters,
        }),
      });
      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'Chat completion failed');
      }

      if (!response.body) {
        throw new Error('Streaming not supported by browser');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let doneEvent = null;

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const parsed = parseNdjsonChunks(buffer);
        buffer = parsed.remainder;

        for (const evt of parsed.events) {
          if (evt.type === 'delta') {
            setMessages((prev) =>
              prev.map((msg, idx) =>
                idx === assistantIndex
                  ? { ...msg, text: `${msg.text || ''}${evt.token || ''}` }
                  : msg,
              ),
            );
          }
          if (evt.type === 'done') {
            doneEvent = evt;
            setMessages((prev) =>
              prev.map((msg, idx) =>
                idx === assistantIndex
                  ? {
                      ...msg,
                      text: evt.answer || msg.text,
                      contexts: Array.isArray(evt.contexts) ? evt.contexts : [],
                      streaming: false,
                    }
                  : msg,
              ),
            );
          }
          if (evt.type === 'error') {
            throw new Error(evt.detail || 'Streaming failed');
          }
        }
      }

      if (buffer.trim()) {
        const evt = JSON.parse(buffer.trim());
        if (evt.type === 'done') {
          doneEvent = evt;
          setMessages((prev) =>
            prev.map((msg, idx) =>
              idx === assistantIndex
                ? {
                    ...msg,
                    text: evt.answer || msg.text,
                    contexts: Array.isArray(evt.contexts) ? evt.contexts : [],
                    streaming: false,
                  }
                : msg,
            ),
          );
        }
      }

      setStatus(`Done. Retrieved ${doneEvent?.contexts?.length || 0} context chunks`);
    } catch (error) {
      setMessages((prev) =>
        prev.map((msg, idx) =>
          idx === assistantIndex
            ? { ...msg, text: `Error: ${error.message}`, streaming: false }
            : msg,
        ),
      );
      setStatus(`Chat error: ${error.message}`);
    } finally {
      setSending(false);
    }
  };

  return (
    <div className="app-shell">
      <header className="hero">
        <h1>RAG Chatbot</h1>
        <p>NotebookLM-style chat over your files, memory, and model answers.</p>
      </header>

      <section className="controls-grid">
        <form className="panel" onSubmit={onChangeConversation}>
          <h2>Conversation</h2>
          <div className="field-row">
            <label htmlFor="convId">ID</label>
            <input
              id="convId"
              value={conversationId}
              onChange={(e) => setConversationId(e.target.value)}
              placeholder="default"
            />
          </div>
          <button type="submit">Load History</button>
        </form>

        <div className="panel">
          <h2>File Upload</h2>
          <input
            type="file"
            multiple
            onChange={(e) => setSelectedFiles(Array.from(e.target.files || []))}
          />
          <button type="button" disabled={!canUpload} onClick={onUpload}>
            {uploading ? 'Uploading...' : `Upload ${selectedFiles.length || ''}`}
          </button>

          <div className="filters">
            <h3>Filter Sources</h3>
            {availableFiles.length === 0 && <p className="hint">No files indexed yet.</p>}
            {availableFiles.map((filename) => (
              <label key={filename} className="filter-item">
                <input
                  type="checkbox"
                  checked={activeFileFilters.includes(filename)}
                  onChange={() => toggleFilter(filename)}
                />
                <span>{filename}</span>
              </label>
            ))}
          </div>
        </div>
      </section>

      <section className="panel chat-panel">
        <h2>Chat</h2>
        <div className="messages">
          {messages.map((msg, idx) => (
            <article className={`message ${msg.role}`} key={`${msg.created_at || 'x'}-${idx}`}>
              <h3>{msg.role === 'user' ? 'You' : 'Assistant'}</h3>
              <p>{msg.text}</p>
              {msg.streaming && <small className="streaming">Streaming...</small>}
              {Array.isArray(msg.contexts) && msg.contexts.length > 0 && (
                <details>
                  <summary>Retrieved Sources ({msg.contexts.length})</summary>
                  {msg.contexts.map((ctx, cIdx) => (
                    <blockquote key={`ctx-${idx}-${cIdx}`}>
                      <strong>[{ctx.citation_id || `S${cIdx + 1}`}]</strong> {ctx.source_label || ctx.kind}
                      <br />
                      {ctx.text}
                    </blockquote>
                  ))}
                </details>
              )}
            </article>
          ))}
        </div>

        <form className="composer" onSubmit={onSend}>
          <textarea
            value={messageInput}
            onChange={(e) => setMessageInput(e.target.value)}
            placeholder="Ask something about your uploaded files and previous messages..."
            rows={4}
          />
          <button type="submit" disabled={!canSend}>
            {sending ? 'Thinking...' : 'Send'}
          </button>
        </form>
      </section>

      <footer className="status-bar">{status}</footer>
    </div>
  );
}

export default App;
