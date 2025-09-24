// App script extracted from templates/index.html and refactored

// Root elements
const hero = document.getElementById('hero');
const heroForm = document.getElementById('hero-form');
const heroInput = document.getElementById('hero-input');
const workspace = document.getElementById('workspace');
const messages = document.getElementById('messages');
const composer = document.getElementById('composer');
const composerInput = document.getElementById('composer-input');
const statusEl = document.getElementById('status');
const notesSearch = document.getElementById('notes-search');
const notesTbody = document.getElementById('notes-tbody');
const btnClearNotes = document.getElementById('clear-notes');

// Summary log elements
const summaryLog = document.getElementById('summary-log');
const summaryLogBody = document.getElementById('summary-log-body');
const summaryActions = document.getElementById('summary-actions');
const btnViewMindmap = document.getElementById('btn-view-mindmap');
let summaryEventSource = null;
let lastMindmapFile = null; // 분석 결과 파일 경로

// Mode toggles
let currentMode = 'ask'; // 'ask' | 'summarize'
const heroModeAskBtn = document.getElementById('hero-mode-ask');
const heroModeSummBtn = document.getElementById('hero-mode-summarize');
const compModeAskBtn = document.getElementById('composer-mode-ask');
const compModeSummBtn = document.getElementById('composer-mode-summarize');

function setMode(newMode) {
  if (newMode !== 'ask' && newMode !== 'summarize') return;
  currentMode = newMode;
  // 활성/비활성 스타일 토글
  [heroModeAskBtn, compModeAskBtn].forEach(b => b && b.classList.toggle('primary', newMode === 'ask'));
  [heroModeSummBtn, compModeSummBtn].forEach(b => b && b.classList.toggle('primary', newMode === 'summarize'));
  // 플레이스홀더 업데이트
  if (newMode === 'ask') {
    heroInput && (heroInput.placeholder = '예: FastAPI에서 파일 업로드는 어떻게 하나요?');
    composerInput && (composerInput.placeholder = '메시지를 입력하세요');
  } else {
    heroInput && (heroInput.placeholder = '예: 아래 텍스트를 요약해줘 (텍스트를 붙여넣으세요)');
    composerInput && (composerInput.placeholder = '요약할 텍스트를 붙여넣으세요');
  }
  // 입력 래퍼 표시/숨김
  const heroAskWrap = document.getElementById('hero-ask-wrap');
  const heroSumWrap = document.getElementById('hero-summarize-wrap');
  if (heroAskWrap && heroSumWrap) {
    heroAskWrap.style.display = (newMode === 'ask') ? '' : 'none';
    heroSumWrap.style.display = (newMode === 'summarize') ? '' : 'none';
  }
  // 채팅 영역은 요약 모드 UI를 제공하지 않으므로 토글하지 않습니다
}

heroModeAskBtn && heroModeAskBtn.addEventListener('click', () => setMode('ask'));
heroModeSummBtn && heroModeSummBtn.addEventListener('click', () => setMode('summarize'));
compModeAskBtn && compModeAskBtn.addEventListener('click', () => setMode('ask'));
compModeSummBtn && compModeSummBtn.addEventListener('click', () => setMode('summarize'));

// Track last user message for assistant memo question autofill
let lastUserText = '';
// Hold latest sources for current assistant answer (memo batch)
let currentSourcesMemo = [];

// Notes storage per conversation
const NotesStore = (() => {
  let cid = null;
  let notes = [];
  function key(c) { return `notes:${c || ''}`; }
  function setCID(newCid) {
    cid = newCid || '';
    notes = load(cid);
    NotesUI.render();
  }
  function load(c) {
    try {
      const raw = localStorage.getItem(key(c));
      return raw ? JSON.parse(raw) : [];
    } catch (_) { return []; }
  }
  function save() {
    try { localStorage.setItem(key(cid), JSON.stringify(notes)); } catch (_) {}
  }
  function nowStr() { return new Date().toISOString(); }
  function add({ kind, question, content, sourceUrl, sources, score, tags }) {
    const note = {
      id: `${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
      cid,
      createdAt: nowStr(),
      kind: kind || 'answer',
      question: (question || '').trim(),
      content: (content || '').trim(),
      sourceUrl: (sourceUrl || '').trim() || undefined,
      sources: Array.isArray(sources) ? sources.map(s => ({
        text: (s && s.text) ? String(s.text) : '',
        url: (s && s.url) ? String(s.url) : undefined
      })) : undefined,
      score: typeof score === 'number' ? score : undefined,
      tags: Array.isArray(tags) ? tags : []
    };
    notes.unshift(note);
    save();
    NotesUI.render();
  }
  function remove(id) {
    notes = notes.filter(n => n.id !== id);
    save();
    NotesUI.render();
  }
  function clear() {
    notes = [];
    save();
    NotesUI.render();
  }
  function list() { return notes.slice(); }
  function update(id, patch) {
    const idx = notes.findIndex(n => n.id === id);
    if (idx === -1) return;
    const base = notes[idx] || {};
    notes[idx] = {
      ...base,
      question: (patch && typeof patch.question === 'string') ? patch.question : base.question,
      content: (patch && typeof patch.content === 'string') ? patch.content : base.content,
      sources: Array.isArray(patch && patch.sources) ? patch.sources : base.sources
    };
    save();
    NotesUI.render();
  }
  function exportJSON() {
    const blob = new Blob([JSON.stringify(notes, null, 2)], { type: 'application/json;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    triggerDownload(`notes_${cid || 'unknown'}.json`, url);
  }
  function exportCSV() {
    const header = ['id','cid','createdAt','kind','question','content','sourceUrl','score','tags'];
    const rows = notes.map(n => [n.id, n.cid, n.createdAt, n.kind, n.question.replaceAll('"', '""'), n.content.replaceAll('"', '""'), n.sourceUrl || '', n.score || '', (n.tags||[]).join('|')]);
    const csv = [header.join(','), ...rows.map(r => `"${r[0]}","${r[1]}","${r[2]}","${r[3]}","${r[4]}","${r[5]}","${r[6]}","${r[7]}","${r[8]}"`)].join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    triggerDownload(`notes_${cid || 'unknown'}.csv`, url);
  }
  function triggerDownload(filename, url) {
    const a = document.createElement('a');
    a.href = url; a.download = filename; a.style.display = 'none';
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 500);
  }
  return { setCID, add, remove, clear, list, exportJSON, exportCSV, update };
})();

// Notes UI rendering and events
const NotesUI = (() => {
  function formatDateYMD(input) {
    try {
      const d = new Date(input);
      if (isNaN(d.getTime())) return String(input || '');
      const y = d.getFullYear();
      const m = String(d.getMonth() + 1).padStart(2, '0');
      const dd = String(d.getDate()).padStart(2, '0');
      return `${y}. ${m}. ${dd}`;
    } catch (_) { return String(input || ''); }
  }
  function render() {
    const all = NotesStore.list();
    const query = (notesSearch.value || '').toLowerCase();
    const data = query ? all.filter(n => (n.question||'').toLowerCase().includes(query) || (n.content||'').toLowerCase().includes(query)) : all;
    notesTbody.innerHTML = '';
    for (const n of data) {
      // main row
      const tr = document.createElement('tr');
      tr.className = 'notes-row';
      const tdDate = document.createElement('td'); tdDate.textContent = formatDateYMD(n.createdAt);
      const tdQ = document.createElement('td'); tdQ.textContent = n.question || '';
      // inline delete button cell
      const tdActions = document.createElement('td');
      const delBtn = document.createElement('button');
      delBtn.className = 'btn';
      delBtn.type = 'button';
      delBtn.textContent = '삭제';
      delBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        if (confirm('이 메모를 삭제하시겠습니까?')) {
          NotesStore.remove(n.id);
        }
      });
      tdActions.appendChild(delBtn);
      tr.appendChild(tdDate); tr.appendChild(tdQ); tr.appendChild(tdActions);

      // toggle expand on click
      tr.addEventListener('click', (event) => {
        const next = tr.nextElementSibling;
        if (next && next.classList.contains('notes-expand')) {
          next.remove();
          return;
        }
        const exp = document.createElement('tr');
        exp.className = 'notes-expand';
          const td = document.createElement('td');
          td.colSpan = 3;
        const inner = document.createElement('div');
        inner.className = 'note-expand-inner';
        const content = document.createElement('pre');
        content.className = 'note-content';
        content.textContent = n.content || '';
        inner.appendChild(content);
        // edit actions
        const actions = document.createElement('div');
        actions.className = 'note-edit-actions';
        const btnEdit = document.createElement('button'); btnEdit.className = 'btn'; btnEdit.type = 'button'; btnEdit.textContent = '수정';
        const btnSave = document.createElement('button'); btnSave.className = 'btn'; btnSave.type = 'button'; btnSave.textContent = '저장'; btnSave.style.display = 'none';
        const btnCancel = document.createElement('button'); btnCancel.className = 'btn'; btnCancel.type = 'button'; btnCancel.textContent = '취소'; btnCancel.style.display = 'none';
        actions.appendChild(btnEdit); actions.appendChild(btnSave); actions.appendChild(btnCancel);
        inner.appendChild(actions);
        actions.addEventListener('click', (e) => e.stopPropagation());
        // sources 배열을 목록으로 표시
        if (Array.isArray(n.sources) && n.sources.length) {
          const list = document.createElement('ul');
          list.className = 'note-sources-list';
          for (const s of n.sources) {
            const li = document.createElement('li');
            if (s && s.url) {
              const a = document.createElement('a');
              a.href = s.url; a.target = '_blank'; a.rel = 'noopener noreferrer';
              a.textContent = s.text || s.url;
              li.appendChild(a);
            } else {
              li.textContent = (s && s.text) ? s.text : '';
            }
            list.appendChild(li);
          }
          inner.appendChild(list);
        }
        td.appendChild(inner);
        exp.appendChild(td);
        tr.insertAdjacentElement('afterend', exp);

        // edit handlers: transform question cell to input, content to textarea
        btnEdit.addEventListener('click', () => {
          const qInput = document.createElement('input');
          qInput.type = 'text'; qInput.className = 'note-input';
          qInput.value = tdQ.textContent || '';
          tdQ.textContent = '';
          tdQ.appendChild(qInput);
          const ta = document.createElement('textarea');
          ta.className = 'note-textarea';
          ta.value = n.content || '';
          content.replaceWith(ta);
          btnEdit.style.display = 'none';
          btnSave.style.display = 'inline-block';
          btnCancel.style.display = 'inline-block';
          qInput.addEventListener('click', (e) => e.stopPropagation());
          ta.addEventListener('click', (e) => e.stopPropagation());
        });
        btnSave.addEventListener('click', () => {
          const newQuestionEl = tdQ.querySelector('input.note-input');
          const newContentEl = tr.nextElementSibling ? tr.nextElementSibling.querySelector('textarea.note-textarea') : null;
          const newQuestion = newQuestionEl ? newQuestionEl.value : (n.question || '');
          const newContent = newContentEl ? newContentEl.value : (n.content || '');
          NotesStore.update(n.id, { question: newQuestion, content: newContent });
        });
        btnCancel.addEventListener('click', () => {
          NotesUI.render();
        });
      });
      notesTbody.appendChild(tr);
    }
  }
  return { render };
})();

function addMessage(text, role) {
  const wrap = document.createElement('div');
  wrap.className = `row ${role}`;
  const col = document.createElement('div');
  col.className = 'message-col';
  wrap.appendChild(col);

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  col.appendChild(bubble);

  // Actions under bubble
  if (role === 'assistant') {
    const actions = document.createElement('div');
    actions.className = 'bubble-actions';
    const memoBtn = document.createElement('button');
    memoBtn.className = 'btn memo-btn';
    memoBtn.type = 'button';
    memoBtn.textContent = '메모';
    memoBtn.addEventListener('click', () => {
      // 클릭 시점의 실제 버블 텍스트와 현재 수집된 출처들을 함께 저장
      const bubbleText = (bubble.textContent || text || '').trim();
      const question = lastUserText || '';
      // 단일 레코드로 답변 + 출처를 함께 저장
      const payloadSources = Array.isArray(currentSourcesMemo) ? currentSourcesMemo.map(it => ({ text: it.text || '', url: it.url || '' })) : [];
      NotesStore.add({ kind: 'answer', question, content: bubbleText, sources: payloadSources });
    });
    actions.appendChild(memoBtn);
    col.appendChild(actions);
  }

  // 출처 UI는 스트리밍 sources 이벤트가 도착했을 때 동적으로 생성합니다.
  messages.appendChild(wrap);
  messages.scrollTop = messages.scrollHeight;
  return bubble;
}

async function ask(question) {
  statusEl.style.display = 'inline';
  const bubble = addMessage('', 'assistant');
  const col = bubble.parentElement; // .message-col
  let sourcesEl = null;
  let sourcesList = null;

  let es;
  try {
    const url = new URL('/ask/stream', window.location.origin);
    url.searchParams.set('q', question);
    // 모드 전달(서버는 현재 무시하지만, 이후 확장 시 활용)
    url.searchParams.set('mode', currentMode);
    const cid = sessionStorage.getItem('conversation_id');
    if (cid) url.searchParams.set('cid', cid);
    es = new EventSource(url.toString());
    // reset sources for this answer
    currentSourcesMemo = [];
    es.addEventListener('token', (e) => {
      bubble.textContent += e.data;
      messages.scrollTop = messages.scrollHeight;
    });
    es.addEventListener('cid', (e) => {
      if (e && e.data) {
        sessionStorage.setItem('conversation_id', e.data);
      }
    });
    es.addEventListener('sources', (e) => {
      try {
        const data = e.data || '[]';
        // Sources render: normalized schema [{type, score, ...}]
        try {
          const payload = JSON.parse(data || '[]');
          const seen = new Set();
          const renderItems = [];
          for (const s of payload) {
            if (!s || typeof s !== 'object') continue;
            if (s.type === 'internal') {
              const q = (s.question || '').trim();
              const ts = (s.timestamp || '').trim();
              const qAuthor = (s.question_author || '').trim();
              // answers / answer_authors: 배열 가정. 혹시 문자열이면 콤마 split
              const answersRaw = s.answers;
              const authorsRaw = s.answer_authors;
              const answers = Array.isArray(answersRaw) ? answersRaw.map(x => String(x).trim()).filter(Boolean) : String(answersRaw || '').split('||').map(x => x.trim()).filter(Boolean);
              let aAuthors = Array.isArray(authorsRaw) ? authorsRaw.map(x => String(x).trim()).filter(Boolean) : String(authorsRaw || '').split('||').map(x => x.trim()).filter(Boolean);
              if (answers.length && aAuthors.length < answers.length) {
                aAuthors = aAuthors.concat(Array(answers.length - aAuthors.length).fill('알 수 없음'));
              }
              const key = `qa::${ts}::${q}`;
              if (seen.has(key)) continue;
              seen.add(key);
              const parts = [];
              if (ts) parts.push(ts);
              if (qAuthor || q) parts.push(`Q) ${qAuthor ? ` ${qAuthor}` : ' : 알 수 없음'} : ${q}`);
              for (let i = 0; i < answers.length; i++) {
                parts.push(`A${i+1}) ${aAuthors[i] || '알 수 없음'} : ${answers[i]}`);
              }
              const text = parts.join('\n');
              renderItems.push({ kind: 'internal', text });
            } else if (s.type === 'web') {
              const urlStr = (s.url || '').trim();
              const title = (s.title || '').trim();
              const key = `web::${urlStr}`;
              if (!urlStr || seen.has(key)) continue;
              seen.add(key);
              renderItems.push({ kind: 'web', text: `${title || urlStr}`, url: urlStr });
            }
          }
          if (renderItems.length) {
            // Save for batch memo
            currentSourcesMemo = renderItems.slice();
            if (!sourcesEl) {
              sourcesEl = document.createElement('details');
              sourcesEl.className = 'sources';
              const summary = document.createElement('summary');
              summary.textContent = '출처';
              sourcesEl.appendChild(summary);
              sourcesList = document.createElement('div');
              sourcesList.className = 'sources-list';
              sourcesEl.appendChild(sourcesList);
              col.appendChild(sourcesEl);
            }
            sourcesList.innerHTML = '';
            for (const it of renderItems) {
              const row = document.createElement('div');
              row.className = 'source-row';
              const span = document.createElement('span');
              span.className = 'source-text';
              // 웹 출처는 링크로 표시
              if (it.url) {
                const a = document.createElement('a');
                a.href = it.url; a.target = '_blank'; a.rel = 'noopener noreferrer';
                a.textContent = it.text;
                span.appendChild(a);
              } else {
                span.textContent = it.text;
              }
              row.appendChild(span);
              sourcesList.appendChild(row);
            }
            sourcesEl.open = true;
          }
        } catch (_) {}
      } catch (_) {}
    });
    es.addEventListener('error', (e) => {
      bubble.textContent += `\n[오류] ${(e && e.data) || ''}`;
    });
    es.addEventListener('done', () => {
      es.close();
      statusEl.style.display = 'none';
    });
  } catch (err) {
    bubble.textContent = `오류: ${err.message || err}`;
    statusEl.style.display = 'none';
  }
}

function validateSummaryInputs({ from, to, email }) {
  return !!(from && to && email);
}

async function handleSendSummaryEmail({ from, to, email, btn }) {
  if (!validateSummaryInputs({ from, to, email })) {
    alert('메일 전송: From/To 날짜와 이메일을 모두 입력해주세요.');
    return;
  }
  const prevText = btn.textContent;
  try {
    statusEl.style.display = 'inline';
    btn.disabled = true;
    btn.classList.add('loading');
    btn.textContent = '보내는 중…';
    // 로그 패널 초기화 및 노출
    if (summaryLog && summaryLogBody) {
      summaryLogBody.textContent = '';
      summaryLog.style.display = '';
    }
    // 마인드맵 버튼 초기화/숨김
    if (summaryActions) summaryActions.style.display = 'none';
    lastMindmapFile = null;
    // 기존 스트림이 있다면 닫기
    if (summaryEventSource) { try { summaryEventSource.close(); } catch (_) {} summaryEventSource = null; }
    // SSE 연결 시도
    let usingSSE = false;
    try {
      const url = new URL('/send-summary-email/stream', window.location.origin);
      url.searchParams.set('from_date', from);
      url.searchParams.set('to_date', to);
      url.searchParams.set('email', email);
      const es = new EventSource(url.toString());
      summaryEventSource = es;
      usingSSE = true;
      es.addEventListener('log', (e) => {
        const line = (e && e.data) ? e.data : '';
        if (line) {
          summaryLogBody.textContent += (summaryLogBody.textContent ? '\n' : '') + line;
          summaryLogBody.scrollTop = summaryLogBody.scrollHeight;
        }
        // 분석 결과 파일 경로 파싱
        try {
          const marker = '분석 결과 저장:';
          const idx = line.indexOf(marker);
          if (idx >= 0) {
            const rest = line.substring(idx + marker.length).trim();
            if (rest && rest.endsWith('.json')) lastMindmapFile = rest;
          }
        } catch (_) {}
      });
      es.addEventListener('error', (e) => {
        const msg = (e && e.data) ? e.data : '알 수 없는 오류';
        summaryLogBody.textContent += (summaryLogBody.textContent ? '\n' : '') + `[오류] ${msg}`;
        summaryLogBody.scrollTop = summaryLogBody.scrollHeight;
      });
      es.addEventListener('done', () => {
        es.close();
        if (summaryEventSource === es) summaryEventSource = null;
        statusEl.style.display = 'none';
        // SSE 완료 시점에 버튼 상태 복구
        btn.disabled = false;
        btn.classList.remove('loading');
        btn.textContent = prevText;
        // 마인드맵 버튼 노출
        if (summaryActions) {
          summaryActions.style.display = '';
          if (btnViewMindmap) {
            btnViewMindmap.onclick = () => {
              const base = '/mindmap';
              const url = (lastMindmapFile && lastMindmapFile.endsWith('.json'))
                ? `${base}?file=${encodeURIComponent(lastMindmapFile)}`
                : base;
              window.location.href = url;
            };
          }
        }
      });
    } catch (err) {
      // SSE 연결 실패 시 폴백: 기존 API 호출
      if (summaryLogBody) summaryLogBody.textContent += (summaryLogBody.textContent ? '\n' : '') + `[오류] 스트리밍 연결 실패: ${(err && err.message) || err}`;
      const res = await fetch('/send-summary-email', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ from_date: from, to_date: to, email })
      });
      if (!res.ok) {
        let errText = '요청 실패';
        try { const er = await res.json(); errText = er.detail || JSON.stringify(er); } catch (_) {}
        alert(errText);
        return;
      }
      const data = await res.json();
      alert((data && data.message) ? data.message : '메일 전송 요청이 접수되었습니다.');
      // 비스트리밍 경로에서는 최신 파일 자동 사용
      if (summaryActions) {
        summaryActions.style.display = '';
        if (btnViewMindmap) {
          btnViewMindmap.onclick = () => { window.location.href = '/mindmap'; };
        }
      }
    }
  } catch (err) {
    alert(`오류: ${(err && err.message) || err}`);
  } finally {
    // 폴백(POST) 경로일 때만 즉시 해제, SSE는 'done'에서 해제
    if (!summaryEventSource) {
      statusEl.style.display = 'none';
      btn.disabled = false;
      btn.classList.remove('loading');
      btn.textContent = prevText;
    }
  }
}

heroForm.addEventListener('submit', (e) => {
  e.preventDefault();
  if (currentMode === 'ask') {
    const q = (heroInput.value || '').trim();
    if (!q) return;
    hero.style.display = 'none';
    workspace.style.display = 'grid';
    addMessage(q, 'user');
    lastUserText = q;
    composerInput.value = '';
    ask(q);
  } else {
    const fromEl = document.getElementById('hero-s-from');
    const toEl = document.getElementById('hero-s-to');
    const emailEl = document.getElementById('hero-s-email');
    const from = fromEl && fromEl.value ? String(fromEl.value) : '';
    const to = toEl && toEl.value ? String(toEl.value) : '';
    const email = emailEl && emailEl.value ? String(emailEl.value).trim() : '';
    if (!from || !to || !email) {
      alert('요약 요청: From/To 날짜와 이메일을 모두 입력해주세요.');
      return;
    }
    // 채팅 화면 전환 없이 히어로 섹션의 메일 전송 버튼을 트리거
    const heroSendBtn = document.getElementById('hero-s-send-email');
    if (heroSendBtn) {
      heroSendBtn.click();
    }
  }
});

composer.addEventListener('submit', (e) => {
  e.preventDefault();
  if (currentMode === 'ask') {
    const q = (composerInput.value || '').trim();
    if (!q) return;
    addMessage(q, 'user');
    lastUserText = q;
    composerInput.value = '';
    ask(q);
  } else {
    alert('요약하기는 메인 화면에서만 가능합니다. 상단 요약 섹션을 이용해주세요.');
  }
});

// 메일보내기 버튼 핸들러 바인딩 (중복 로직 통합)
(function bindSendEmail(){
  const heroBtn = document.getElementById('hero-s-send-email');
  if (heroBtn) heroBtn.addEventListener('click', async () => {
    const fromEl = document.getElementById('hero-s-from');
    const toEl = document.getElementById('hero-s-to');
    const emailEl = document.getElementById('hero-s-email');
    const from = fromEl && fromEl.value ? String(fromEl.value) : '';
    const to = toEl && toEl.value ? String(toEl.value) : '';
    const email = emailEl && emailEl.value ? String(emailEl.value).trim() : '';
    await handleSendSummaryEmail({ from, to, email, btn: heroBtn });
  });
})();

// Notes toolbar events
notesSearch.addEventListener('input', () => NotesUI.render());
btnClearNotes.addEventListener('click', () => NotesStore.clear());

// Sync notes with conversation_id when received
window.addEventListener('storage', (e) => {
  if (e.key && e.key.startsWith('notes:')) NotesUI.render();
});

// When CID is set by server, update store
(function observeCID() {
  const origSetItem = sessionStorage.setItem.bind(sessionStorage);
  sessionStorage.setItem = function(k, v) {
    origSetItem(k, v);
    if (k === 'conversation_id') {
      NotesStore.setCID(v);
    }
  };
  const existing = sessionStorage.getItem('conversation_id');
  if (existing) NotesStore.setCID(existing);
})();

// 새 채팅(new=1) 진입 시 클라이언트 상태 초기화
(function resetOnNewChat() {
  try {
    const sp = new URLSearchParams(window.location.search);
    if (sp.get('new') === '1') {
      // 대화 식별자 초기화 -> 서버 측 도메인 메모리도 새로운 CID로 분리됨
      sessionStorage.removeItem('conversation_id');
      // UI 초기화
      if (messages) messages.innerHTML = '';
      if (hero) hero.style.display = '';
      if (workspace) workspace.style.display = 'none';
      if (summaryLogBody) summaryLogBody.textContent = '';
      if (summaryLog) summaryLog.style.display = 'none';
      if (summaryActions) summaryActions.style.display = 'none';
      lastUserText = '';
      currentSourcesMemo = [];
      // 메모 패널도 현재 CID 기준으로 비어있는 상태 렌더
      NotesStore.setCID('');
    }
  } catch (_) {}
})();


