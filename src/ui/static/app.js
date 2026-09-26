/**
 * Unified Voice Agent Workbench - Client Application
 * Handles session orchestration, real-time message streaming,
 * and live diagnostics telemetry.
 */

(function () {
  'use strict';

  // --- State Variables ---
  let activeSessionId = null;
  let turnCount = 0;
  let isProcessing = false;

  // --- DOM Elements ---
  const callDirectionSelect = document.getElementById('call-direction');
  const outboundFields = document.getElementById('outbound-fields');
  const domainSelect = document.getElementById('domain-select');
  const callerNameInput = document.getElementById('caller-name');
  const callerPhoneInput = document.getElementById('caller-phone');
  const callerEmailInput = document.getElementById('caller-email');
  const callerCompanyInput = document.getElementById('caller-company');
  const campaignObjectiveInput = document.getElementById('campaign-objective');
  const knownPurposeInput = document.getElementById('known-purpose');

  const btnStart = document.getElementById('btn-start');
  const btnReset = document.getElementById('btn-reset');
  const chatMessages = document.getElementById('chat-messages');
  const emptyState = document.getElementById('empty-state');
  const chatForm = document.getElementById('chat-form');
  const userInput = document.getElementById('user-input');
  const btnSend = document.getElementById('btn-send');
  const quickPrompts = document.getElementById('quick-prompts');

  // Badges & Meta
  const chatSessionTitle = document.getElementById('chat-session-title');
  const chatDirectionBadge = document.getElementById('chat-direction-badge');
  const chatDomainBadge = document.getElementById('chat-domain-badge');
  const sessionIdDisplay = document.getElementById('session-id-display');
  const sessionStatusBadge = document.getElementById('session-status-badge');
  const turnCounter = document.getElementById('turn-counter');
  const errorBanner = document.getElementById('error-banner');
  const errorText = document.getElementById('error-text');

  // Latency Elements
  const latTotal = document.getElementById('lat-total');
  const latPrompt = document.getElementById('lat-prompt');
  const latDecision = document.getElementById('lat-decision');
  const latRag = document.getElementById('lat-rag');
  const latGrounded = document.getElementById('lat-grounded');
  const latTool = document.getElementById('lat-tool');

  // Intent & State Elements
  const diagDomain = document.getElementById('diag-domain');
  const diagIntent = document.getElementById('diag-intent');
  const diagSubIntent = document.getElementById('diag-sub-intent');
  const diagStage = document.getElementById('diag-stage');
  const diagPendingQ = document.getElementById('diag-pending-q');
  const diagPendingA = document.getElementById('diag-pending-a');

  // Entities, RAG, MCP
  const diagSlotsContainer = document.getElementById('diag-slots-container');
  const diagRagRequired = document.getElementById('diag-rag-required');
  const diagRagQuery = document.getElementById('diag-rag-query');
  const diagCitationsList = document.getElementById('diag-citations-list');
  const diagActionProposed = document.getElementById('diag-action-proposed');
  const diagToolName = document.getElementById('diag-tool-name');
  const diagToolJson = document.getElementById('diag-tool-json');

  // System Badges
  const badgeLlm = document.getElementById('badge-llm');
  const badgeRag = document.getElementById('badge-rag');
  const badgeTool = document.getElementById('badge-tool');
  const badgeStatus = document.getElementById('badge-status');
  const systemNotesList = document.getElementById('system-notes-list');

  // --- Initial Setup & Listeners ---
  function init() {
    setupDirectionToggle();
    setupEventListeners();
    fetchSystemStatus();
  }

  function setupDirectionToggle() {
    function updateVisibility() {
      const isOutbound = callDirectionSelect.value === 'outbound';
      if (outboundFields) {
        outboundFields.style.display = isOutbound ? 'block' : 'none';
      }
    }
    callDirectionSelect.addEventListener('change', updateVisibility);
    updateVisibility();
  }

  function setupEventListeners() {
    btnStart.addEventListener('click', handleStartSession);
    btnReset.addEventListener('click', handleResetSession);
    chatForm.addEventListener('submit', handleSendMessage);

    // Quick prompt buttons
    if (quickPrompts) {
      quickPrompts.addEventListener('click', (e) => {
        const btn = e.target.closest('.quick-btn');
        if (!btn || isProcessing) return;
        const text = btn.dataset.text;
        if (text && activeSessionId) {
          userInput.value = text;
          handleSendMessage(new Event('submit'));
        }
      });
    }
  }

  // --- API Calls ---
  async function fetchSystemStatus() {
    try {
      const res = await fetch('/api/status');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      if (badgeLlm) badgeLlm.textContent = `LLM: ${data.llm_mode || 'Gemini'}`;
      if (badgeRag) badgeRag.textContent = `RAG: ${data.rag_mode || 'Hybrid'}`;
      if (badgeTool) badgeTool.textContent = `MCP: ${data.tool_mode || 'Active'}`;
      if (badgeStatus) {
        badgeStatus.textContent = data.overall_mode === 'LIVE' ? 'LIVE' : data.overall_mode;
        badgeStatus.className = `badge badge-status ${data.overall_mode === 'LIVE' ? 'status-live' : 'status-ready'}`;
      }

      if (systemNotesList && data.initialization_notes) {
        systemNotesList.innerHTML = '';
        data.initialization_notes.forEach((note) => {
          const li = document.createElement('li');
          li.textContent = note;
          systemNotesList.appendChild(li);
        });
      }
    } catch (err) {
      console.warn('Could not fetch status:', err);
      showError(`Status check failed: ${err.message}`);
    }
  }

  async function handleStartSession() {
    if (isProcessing) return;
    hideError();

    const direction = callDirectionSelect.value;
    const domain = domainSelect.value;
    const callerName = callerNameInput.value.trim() || 'Valued Caller';
    const callerPhone = callerPhoneInput.value.trim() || '+15551234567';
    const callerEmail = callerEmailInput.value.trim() || 'caller@example.com';
    const callerCompany = callerCompanyInput.value.trim() || 'Acme Corp';
    const campaignObjective = campaignObjectiveInput.value.trim() || 'Product follow-up consultation';
    const knownPurpose = knownPurposeInput.value.trim() || 'AI consulting services';

    const payload = {
      direction: direction,
      domain: domain,
      caller_name: callerName,
      caller_phone: callerPhone,
      caller_email: callerEmail,
      caller_company: callerCompany,
      campaign_objective: campaignObjective,
      known_purpose: knownPurpose,
    };

    setProcessing(true, 'Starting session...');

    try {
      const res = await fetch('/api/sessions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server returned ${res.status}`);
      }

      const data = await res.json();
      activeSessionId = data.session_id;
      turnCount = 0;

      // Update UI Header
      chatSessionTitle.textContent = `${direction.toUpperCase()} Call Active`;
      chatDirectionBadge.textContent = direction.toUpperCase();
      chatDirectionBadge.className = `badge badge-${direction}`;
      chatDomainBadge.textContent = domain.toUpperCase();
      sessionIdDisplay.textContent = `Session: ${activeSessionId}`;
      sessionStatusBadge.textContent = 'ACTIVE';
      sessionStatusBadge.className = 'meta-badge status-active';

      // Clear chat messages and hide empty state
      chatMessages.innerHTML = '';
      if (emptyState) emptyState.remove();

      // Enable Chat Input
      userInput.disabled = false;
      btnSend.disabled = false;
      userInput.focus();

      // If Outbound, the agent speaks first!
      if (direction === 'outbound' && data.opening_message) {
        appendMessage('agent', data.opening_message);
        turnCount = 1;
        turnCounter.textContent = `Turn ${turnCount}`;
      } else {
        // Inbound: Show hint that agent is waiting for caller
        const hint = document.createElement('div');
        hint.className = 'chat-notice';
        hint.innerHTML = '<em>Inbound call connected. Waiting for caller to speak first...</em>';
        chatMessages.appendChild(hint);
      }

      if (data.diagnostics) {
        updateDiagnostics(data.diagnostics);
      }
    } catch (err) {
      showError(`Failed to start session: ${err.message}`);
    } finally {
      setProcessing(false);
    }
  }

  async function handleSendMessage(e) {
    if (e && e.preventDefault) e.preventDefault();
    if (isProcessing || !activeSessionId) return;

    const message = userInput.value.trim();
    if (!message) return;

    // Clear input
    userInput.value = '';
    hideError();

    // Render User Message
    appendMessage('user', message);
    turnCount += 1;
    turnCounter.textContent = `Turn ${turnCount}`;

    setProcessing(true, 'Agent thinking...');

    // Render loading agent bubble
    const loadingBubble = appendLoadingBubble();

    try {
      const res = await fetch(`/api/sessions/${activeSessionId}/turns`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_message: message }),
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || `Server returned ${res.status}`);
      }

      const data = await res.json();

      // Remove loading indicator and replace with actual response
      if (loadingBubble && loadingBubble.parentNode) {
        loadingBubble.remove();
      }

      appendMessage('agent', data.response_text);

      if (data.diagnostics) {
        updateDiagnostics(data.diagnostics);
      }
    } catch (err) {
      if (loadingBubble && loadingBubble.parentNode) {
        loadingBubble.remove();
      }
      showError(`Turn execution error: ${err.message}`);
      appendMessage('agent', 'I encountered an issue processing that. Please try again.');
    } finally {
      setProcessing(false);
      userInput.focus();
    }
  }

  async function handleResetSession() {
    if (!activeSessionId) {
      resetUI();
      return;
    }

    try {
      await fetch(`/api/sessions/${activeSessionId}/reset`, { method: 'POST' });
    } catch (err) {
      console.warn('Reset warning:', err);
    }

    resetUI();
  }

  function resetUI() {
    activeSessionId = null;
    turnCount = 0;
    isProcessing = false;

    chatSessionTitle.textContent = 'No Active Call';
    chatDirectionBadge.textContent = 'INBOUND';
    chatDirectionBadge.className = 'badge badge-outline';
    chatDomainBadge.textContent = 'VAYVORA';
    sessionIdDisplay.textContent = 'Session: —';
    sessionStatusBadge.textContent = 'IDLE';
    sessionStatusBadge.className = 'meta-badge status-idle';
    turnCounter.textContent = 'Turn 0';

    chatMessages.innerHTML = `
      <div class="empty-state" id="empty-state">
        <div class="empty-icon">🎧</div>
        <h3>Call Workbench Idle</h3>
        <p>Select your direction and domain, then click <strong>Start Session</strong> to initiate testing.</p>
      </div>
    `;

    userInput.value = '';
    userInput.disabled = true;
    btnSend.disabled = true;

    // Reset diagnostics
    resetDiagnostics();
    hideError();
  }

  // --- DOM Helpers ---
  function appendMessage(sender, text) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `chat-message msg-${sender}`;

    const avatar = document.createElement('div');
    avatar.className = 'msg-avatar';
    avatar.textContent = sender === 'user' ? '👤' : '🤖';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'msg-content';

    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';
    bubble.textContent = text;

    const time = document.createElement('span');
    time.className = 'msg-timestamp';
    time.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

    contentDiv.appendChild(bubble);
    contentDiv.appendChild(time);
    msgDiv.appendChild(avatar);
    msgDiv.appendChild(contentDiv);

    chatMessages.appendChild(msgDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return msgDiv;
  }

  function appendLoadingBubble() {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'chat-message msg-agent msg-loading';

    const avatar = document.createElement('div');
    avatar.className = 'msg-avatar';
    avatar.textContent = '🤖';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'msg-content';

    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble loading-dots';
    bubble.innerHTML = '<span>.</span><span>.</span><span>.</span>';

    contentDiv.appendChild(bubble);
    msgDiv.appendChild(avatar);
    msgDiv.appendChild(contentDiv);

    chatMessages.appendChild(msgDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return msgDiv;
  }

  function updateDiagnostics(diag) {
    if (!diag) return;

    // 1. Latency Breakdown
    if (diag.latency) {
      const lat = diag.latency;
      const total = lat.total_turn_ms !== null && lat.total_turn_ms !== undefined ? Number(lat.total_turn_ms).toFixed(1) : '—';
      latTotal.textContent = total;

      const bk = lat.breakdown || {};
      latPrompt.textContent = bk.prompt_construction_ms ? `${Number(bk.prompt_construction_ms).toFixed(1)}ms` : '—';
      latDecision.textContent = bk.llm_decision_ms ? `${Number(bk.llm_decision_ms).toFixed(1)}ms` : '—';
      latRag.textContent = bk.rag_retrieval_ms ? `${Number(bk.rag_retrieval_ms).toFixed(1)}ms` : '—';
      latGrounded.textContent = bk.grounded_llm_ms ? `${Number(bk.grounded_llm_ms).toFixed(1)}ms` : '—';
      latTool.textContent = bk.tool_execution_ms ? `${Number(bk.tool_execution_ms).toFixed(1)}ms` : '—';
    }

    // 2. Intent & State Progression
    if (diag.session) {
      diagDomain.textContent = (diag.session.company || '—').toUpperCase();
      diagStage.textContent = diag.session.conversation_stage || '—';
    }

    if (diag.intent) {
      diagIntent.textContent = diag.intent.current_intent || '—';
      diagSubIntent.textContent = diag.intent.sub_intent || '—';
    }

    if (diag.entities) {
      diagPendingQ.textContent = diag.entities.pending_question || 'None';

      // Extracted Slots
      diagSlotsContainer.innerHTML = '';
      const slots = diag.entities.extracted_slots || {};
      const slotKeys = Object.keys(slots);
      if (slotKeys.length === 0) {
        diagSlotsContainer.innerHTML = '<span class="empty-tag">No entity slots extracted yet</span>';
      } else {
        slotKeys.forEach((key) => {
          const tag = document.createElement('span');
          tag.className = 'slot-tag';
          tag.textContent = `${key}: ${slots[key]}`;
          diagSlotsContainer.appendChild(tag);
        });
      }
    }

    // 3. Grounded RAG Status
    if (diag.rag) {
      diagRagRequired.textContent = diag.rag.knowledge_required ? 'True' : 'False';
      diagRagRequired.className = `diag-value ${diag.rag.knowledge_required ? 'val-true' : 'val-false'}`;
      diagRagQuery.textContent = diag.rag.knowledge_query || '—';

      diagCitationsList.innerHTML = '';
      const citations = diag.rag.grounded_citations || [];
      if (citations.length === 0) {
        diagCitationsList.innerHTML = '<li class="empty-item">None for current turn</li>';
      } else {
        citations.forEach((c) => {
          const li = document.createElement('li');
          li.className = 'citation-item';
          li.textContent = typeof c === 'string' ? c : JSON.stringify(c);
          diagCitationsList.appendChild(li);
        });
      }
    }

    // 4. MCP Tools
    if (diag.tools) {
      const isProposed = Boolean(diag.tools.proposed_action);
      diagActionProposed.textContent = isProposed ? 'True' : 'False';
      diagActionProposed.className = `diag-value ${isProposed ? 'val-true' : 'val-false'}`;

      if (isProposed && diag.tools.proposed_action.tool_name) {
        diagToolName.textContent = diag.tools.proposed_action.tool_name;
      } else if (diag.tools.last_action) {
        diagToolName.textContent = `${diag.tools.last_action} (${diag.tools.verification_status || 'done'})`;
      } else {
        diagToolName.textContent = '—';
      }

      diagPendingA.textContent = diag.tools.tool_status === 'pending' ? 'Pending Confirmation' : 'None';

      // JSON Representation
      const toolPayload = {
        status: diag.tools.tool_status,
        verification: diag.tools.verification_status,
        proposed: diag.tools.proposed_action,
        reference: diag.tools.external_reference,
        failure: diag.tools.failure_reason,
      };
      diagToolJson.textContent = JSON.stringify(toolPayload, null, 2);
    }
  }

  function resetDiagnostics() {
    latTotal.textContent = '0.0';
    latPrompt.textContent = '—';
    latDecision.textContent = '—';
    latRag.textContent = '—';
    latGrounded.textContent = '—';
    latTool.textContent = '—';

    diagDomain.textContent = '—';
    diagIntent.textContent = '—';
    diagSubIntent.textContent = '—';
    diagStage.textContent = '—';
    diagPendingQ.textContent = '—';
    diagPendingA.textContent = '—';

    diagSlotsContainer.innerHTML = '<span class="empty-tag">No entity slots extracted yet</span>';
    diagRagRequired.textContent = 'False';
    diagRagQuery.textContent = '—';
    diagCitationsList.innerHTML = '<li class="empty-item">None for current turn</li>';

    diagActionProposed.textContent = 'False';
    diagToolName.textContent = '—';
    diagToolJson.textContent = 'No tool execution recorded yet.';
  }

  function setProcessing(loading, statusText) {
    isProcessing = loading;
    if (btnSend) btnSend.disabled = loading;
    if (userInput && !activeSessionId) userInput.disabled = true;

    if (badgeStatus) {
      if (loading) {
        badgeStatus.textContent = statusText || 'Processing...';
        badgeStatus.className = 'badge badge-status status-working';
      } else {
        badgeStatus.textContent = 'Ready';
        badgeStatus.className = 'badge badge-status status-ready';
      }
    }
  }

  function showError(msg) {
    if (errorBanner && errorText) {
      errorText.textContent = msg;
      errorBanner.classList.remove('hidden');
    }
  }

  function hideError() {
    if (errorBanner) {
      errorBanner.classList.add('hidden');
    }
  }

  // --- Run on Load ---
  document.addEventListener('DOMContentLoaded', init);
})();
