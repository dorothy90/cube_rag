(function() {
  const svg = document.getElementById('mindmap');
  const tooltip = document.getElementById('tooltip');
  // 수동 입력 UI 제거: 버튼/입력 대신 URL 쿼리의 file만 사용
  const btnLoad = null;
  const btnReset = null;
  const fileInput = null;

  const d3sel = d3.select(svg);
  const width = () => svg.clientWidth || 800;
  const height = () => svg.clientHeight || 600;

  let gRoot = d3sel.append('g');
  let zoom = d3.zoom().scaleExtent([0.3, 2]).on('zoom', (event) => {
    gRoot.attr('transform', event.transform);
  });
  d3sel.call(zoom);

  function clear() {
    gRoot.selectAll('*').remove();
  }

  async function fetchData() {
    // URL 쿼리에서 file만 수용
    let file = null;
    try {
      const params = new URLSearchParams(window.location.search || '');
      const qFile = params.get('file');
      if (qFile) file = qFile;
    } catch (e) {}
    const url = new URL('/api/mindmap-data', window.location.origin);
    if (file) url.searchParams.set('file', file);
    const res = await fetch(url.toString());
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(txt || '로드 실패');
    }
    const data = await res.json();
    if (!data || !data.tree) throw new Error('유효하지 않은 데이터');
    return data.tree;
  }

  function render(treeData) {
    clear();
    const root = d3.hierarchy(treeData);
    const layout = d3.tree().nodeSize([32, 200]);
    layout(root);

    // 중앙 정렬을 위한 translate
    const x0 = d3.min(root.descendants(), d => d.x) || 0;
    const x1 = d3.max(root.descendants(), d => d.x) || 0;
    const y1 = d3.max(root.descendants(), d => d.y) || 0;
    const offX = (height() - (x1 - x0)) / 2;
    const offY = 40;
    const container = gRoot.append('g').attr('transform', `translate(${offY},${offX - x0})`);

    // Links
    container.selectAll('path.link')
      .data(root.links())
      .enter()
      .append('path')
      .attr('class', 'link')
      .attr('fill', 'none')
      .attr('stroke', '#aaa')
      .attr('stroke-width', 1.5)
      .attr('d', d3.linkHorizontal().x(d => d.y).y(d => d.x));

    // Nodes
    const node = container.selectAll('g.node')
      .data(root.descendants())
      .enter()
      .append('g')
      .attr('class', d => 'node' + (d.depth === 0 ? ' root' : (d.children ? ' parent' : ' leaf')))
      .attr('transform', d => `translate(${d.y},${d.x})`);

    node.append('circle')
      .attr('r', d => d.depth === 0 ? 8 : (d.children ? 6 : 4))
      .attr('fill', d => d.depth === 0 ? '#0ea5e9' : (d.children ? '#22c55e' : '#6366f1'));

    node.append('text')
      .attr('dy', '0.32em')
      .attr('x', d => d.children ? -10 : 10)
      .attr('text-anchor', d => d.children ? 'end' : 'start')
      .style('font-size', '12px')
      .text(d => (d.data && d.data.name) ? d.data.name : '(이름없음)')
      .style('cursor', d => (!d.children && d.depth > 0) ? 'pointer' : 'default')
      .on('click', async function(event, d) {
        // leaf(소주제) 노드만 동작
        if (d.children || d.depth === 0) return;
        const meta = (d && d.data && d.data.meta) ? d.data.meta : null;
        if (!meta || typeof meta.subtopic_index === 'undefined') return;
        const topicIdx = (typeof meta.topic_index === 'number') ? meta.topic_index : null;
        const subIdx = meta.subtopic_index;
        try {
          await loadMessages(topicIdx, subIdx);
        } catch (err) {
          alert(`메시지 로드 오류: ${(err && err.message) || err}`);
        }
      })
      .on('mousemove', function(event, d) {
        const meta = (d && d.data && d.data.meta) ? d.data.meta : null;
        const lines = [];
        if (meta) {
          if (typeof meta.subtopic_index !== 'undefined') lines.push(`#${meta.subtopic_index}`);
          if (meta.summary) lines.push(meta.summary);
          if (Array.isArray(meta.keywords) && meta.keywords.length) lines.push(`키워드: ${meta.keywords.join(', ')}`);
          if (typeof meta.message_count === 'number') lines.push(`관련 메시지 수: ${meta.message_count}`);
        }
        if (lines.length) {
          tooltip.style.display = 'block';
          tooltip.textContent = lines.join('\n');
          const { pageX, pageY } = event;
          const rect = svg.getBoundingClientRect();
          tooltip.style.left = (pageX - rect.left + 12) + 'px';
          tooltip.style.top = (pageY - rect.top + 12) + 'px';
        }
      })
      .on('mouseleave', function() {
        tooltip.style.display = 'none';
      });
  }

  async function load() {
    try {
      const data = await fetchData();
      render(data);
    } catch (err) {
      alert(`오류: ${(err && err.message) || err}`);
    }
  }

  async function loadMessages(topicIndex, subtopicIndex) {
    if (typeof topicIndex !== 'number') return;
    let file = null;
    try {
      const params = new URLSearchParams(window.location.search || '');
      const qFile = params.get('file');
      if (qFile) file = qFile;
    } catch (e) {}
    const url = new URL('/api/mindmap-messages', window.location.origin);
    url.searchParams.set('topic_index', String(topicIndex));
    url.searchParams.set('subtopic_index', String(subtopicIndex));
    if (file) url.searchParams.set('file', file);
    const res = await fetch(url.toString());
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(txt || '메시지 로드 실패');
    }
    const payload = await res.json();
    const listEl = document.getElementById('messages-list');
    const panel = document.getElementById('messages-panel');
    const titleEl = document.getElementById('messages-title');
    if (!listEl || !panel || !titleEl) return;
    listEl.innerHTML = '';
    const subName = (payload && payload.subtopic && payload.subtopic.name) ? payload.subtopic.name : '';
    titleEl.textContent = subName ? `관련 메시지 · ${subName}` : '관련 메시지';
    const msgs = (payload && Array.isArray(payload.messages)) ? payload.messages : [];
    for (const m of msgs) {
      const row = document.createElement('div');
      row.className = 'chat-row';
      const col = document.createElement('div');
      const meta = document.createElement('div'); meta.className = 'meta';
      const idStr = (m && m.message_id) ? `#${m.message_id}` : '';
      const reacts = (typeof m.reaction_count === 'number') ? ` · 반응 ${m.reaction_count}` : '';
      meta.textContent = `${idStr}${reacts}`;
      const bubble = document.createElement('div'); bubble.className = 'bubble';
      bubble.textContent = (m && m.content) ? String(m.content) : '';
      col.appendChild(meta);
      col.appendChild(bubble);
      row.appendChild(col);
      listEl.appendChild(row);
    }
    panel.style.display = '';
  }

  // 수동 버튼 제거됨

  load();
})();


