/* lab.js — the evaluation lab, a theme toggle and the contents highlight.
   Model: f(x) = sigmoid(S * (w.x + b)). S = 1 for the standard model, 200 for the saturated-output model.
   Both give identical decisions. They differ only in the gradient a naive attack can read:
   probabilities are rounded to float32, so a saturated output returns p = 1 exactly and 1 - p = 0. */
(() => {
  /* ---------- theme + contents ---------- */
  const root = document.documentElement, tb = document.getElementById('theme'), tt = document.getElementById('themeTxt');
  const sys = () => matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  const cur = () => root.getAttribute('data-theme') || sys();
  const paintTheme = () => { tt.textContent = cur() === 'dark' ? 'Dark' : 'Light'; };
  tb.addEventListener('click', () => { root.setAttribute('data-theme', cur() === 'dark' ? 'light' : 'dark'); paintTheme(); requestAnimationFrame(() => redraw()); });
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { paintTheme(); requestAnimationFrame(() => requestAnimationFrame(() => redraw())); });
  paintTheme();

  const links = [...document.querySelectorAll('.toc a')], secs = links.map(a => document.querySelector(a.getAttribute('href')));
  const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) links.forEach(a => a.classList.toggle('on', a.getAttribute('href') === '#' + e.target.id)); }), { rootMargin: '-35% 0px -60% 0px' });
  secs.forEach(s => s && io.observe(s));

  /* ---------- data: two Gaussian classes, fixed seed ---------- */
  let seed = 20260;
  const rnd = () => (seed = (seed * 1664525 + 1013904223) % 4294967296) / 4294967296;
  const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(6.2832 * rnd());
  const pts = [];
  for (let i = 0; i < 90; i++) { const y = i % 2 ? 1 : -1; pts.push({ x: [y * 1.0 + gauss() * .62, y * .55 + gauss() * .62], y }); }

  /* ---------- train logistic regression on the standard scale ---------- */
  let w = [0, 0], b = 0;
  for (let it = 0; it < 600; it++) {
    let gw0 = 0, gw1 = 0, gb = 0;
    pts.forEach(p => { const z = w[0] * p.x[0] + w[1] * p.x[1] + b, t = p.y > 0 ? 1 : 0, e = 1 / (1 + Math.exp(-z)) - t; gw0 += e * p.x[0]; gw1 += e * p.x[1]; gb += e; });
    w[0] -= .3 * gw0 / pts.length; w[1] -= .3 * gw1 / pts.length; b -= .3 * gb / pts.length;
  }
  const logit = x => w[0] * x[0] + w[1] * x[1] + b;
  const predict = x => logit(x) > 0 ? 1 : -1;

  /* ---------- attacks ---------- */
  const sig = z => Math.fround(1 / (1 + Math.exp(-z)));
  const sgn = v => v > 0 ? 1 : v < 0 ? -1 : 0;
  function attack(p, kind, eps, S) {
    if (kind === 'noise') return p.x.map((v, i) => v + eps * (rnd() * 2 - 1));
    // Loss L = -log p_y. dL/dx = -(1 - p_y) * S * y * w, with p_y computed in float32.
    const py = sig(S * p.y * logit(p.x)), oneMinus = Math.fround(1 - py);
    const g = kind === 'naive' ? w.map(wi => -oneMinus * S * p.y * wi) : w.map(wi => -p.y * wi); // adaptive: gradient of the logit itself
    return p.x.map((v, i) => v + eps * sgn(g[i]));
  }
  const accuracy = (kind, eps, S) => {
    seed = 777; let ok = 0;
    pts.forEach(p => { if (predict(attack(p, kind, eps, S)) === p.y) ok++; });
    return ok / pts.length;
  };

  /* ---------- drawing ---------- */
  const $ = id => document.getElementById(id);
  const plane = $('cvPlane'), curve = $('cvCurve'), pc = plane.getContext('2d'), cc = curve.getContext('2d');
  const css = n => getComputedStyle(root).getPropertyValue(n).trim();
  const X0 = -3, X1 = 3, Y0 = -2.2, Y1 = 2.2;
  const sx = x => (x - X0) / (X1 - X0) * plane.width, sy = y => plane.height - (y - Y0) / (Y1 - Y0) * plane.height;
  const COLORS = { noise: () => css('--muted'), naive: () => css('--accent'), adaptive: () => css('--attack') };
  const NAMES = { noise: 'Random noise', naive: 'Naive gradient attack', adaptive: 'Adaptive attack' };
  const read = () => ({ model: document.querySelector('input[name=model]:checked').value, attack: document.querySelector('input[name=attack]:checked').value, eps: +$('eps').value });

  function drawPlane(st, S) {
    const W = plane.width, H = plane.height, ink = css('--ink'), mu = css('--muted'), line = css('--line');
    pc.clearRect(0, 0, W, H); pc.font = '12px "IBM Plex Mono", monospace';
    // decision regions
    pc.fillStyle = css('--accent-soft'); pc.beginPath();
    // region where predict == +1 : w.x + b > 0, clipped to the box
    const poly = [[X0, Y0], [X1, Y0], [X1, Y1], [X0, Y1]];
    const clip = (P, f) => { const out = []; for (let i = 0; i < P.length; i++) { const a = P[i], c = P[(i + 1) % P.length], fa = f(a), fc = f(c); if (fa >= 0) out.push(a); if ((fa >= 0) !== (fc >= 0)) { const t = fa / (fa - fc); out.push([a[0] + t * (c[0] - a[0]), a[1] + t * (c[1] - a[1])]); } } return out; };
    const pos = clip(poly, a => logit(a));
    pos.forEach((a, i) => i ? pc.lineTo(sx(a[0]), sy(a[1])) : pc.moveTo(sx(a[0]), sy(a[1]))); pc.fill();
    // boundary
    pc.strokeStyle = mu; pc.lineWidth = 1.5; pc.setLineDash([6, 4]); pc.beginPath();
    const yb = x => -(w[0] * x + b) / w[1]; pc.moveTo(sx(X0), sy(yb(X0))); pc.lineTo(sx(X1), sy(yb(X1))); pc.stroke(); pc.setLineDash([]);
    // points
    seed = 777;
    pts.forEach(p => {
      const q = attack(p, st.attack, st.eps, S), flipped = predict(q) !== p.y, col = p.y > 0 ? css('--accent') : css('--ink');
      if (st.eps > 0) { pc.strokeStyle = line; pc.lineWidth = 1; pc.beginPath(); pc.moveTo(sx(p.x[0]), sy(p.x[1])); pc.lineTo(sx(q[0]), sy(q[1])); pc.stroke(); }
      pc.fillStyle = col; pc.globalAlpha = .35; pc.beginPath(); pc.arc(sx(p.x[0]), sy(p.x[1]), 3.5, 0, 6.2832); pc.fill(); pc.globalAlpha = 1;
      pc.fillStyle = col; pc.beginPath(); pc.arc(sx(q[0]), sy(q[1]), 4.5, 0, 6.2832); pc.fill();
      if (flipped) { pc.strokeStyle = css('--attack'); pc.lineWidth = 2; pc.beginPath(); pc.arc(sx(q[0]), sy(q[1]), 9, 0, 6.2832); pc.stroke(); }
    });
    pc.fillStyle = mu; pc.fillText('x₁', W - 24, H - 10); pc.fillText('x₂', 8, 16);
  }

  function drawCurve(st, S) {
    const W = curve.width, H = curve.height, m = { l: 52, r: 18, t: 18, b: 44 }, pw = W - m.l - m.r, ph = H - m.t - m.b;
    const ink = css('--ink'), mu = css('--muted'), line = css('--line'), EMAX = 1.5;
    const px = e => m.l + e / EMAX * pw, py = a => m.t + (1 - a) * ph;
    cc.clearRect(0, 0, W, H); cc.font = '12px "IBM Plex Mono", monospace'; cc.textBaseline = 'middle';
    cc.strokeStyle = line; cc.fillStyle = mu; cc.lineWidth = 1;
    [0, .25, .5, .75, 1].forEach(a => { cc.beginPath(); cc.moveTo(m.l, py(a)); cc.lineTo(W - m.r, py(a)); cc.stroke(); cc.textAlign = 'right'; cc.fillText(Math.round(a * 100) + '%', m.l - 8, py(a)); });
    cc.textAlign = 'center'; [0, .5, 1, 1.5].forEach(e => { cc.fillText(e.toFixed(1), px(e), H - m.b + 18); });
    cc.fillText('attack budget ε', m.l + pw / 2, H - 8); cc.save(); cc.translate(14, m.t + ph / 2); cc.rotate(-Math.PI / 2); cc.fillText('accuracy under attack', 0, 0); cc.restore();
    ['noise', 'naive', 'adaptive'].forEach(k => {
      cc.strokeStyle = COLORS[k](); cc.lineWidth = k === st.attack ? 3 : 1.6; cc.globalAlpha = k === st.attack ? 1 : .65; cc.beginPath();
      for (let i = 0; i <= 60; i++) { const e = i / 60 * EMAX, a = accuracy(k, e, S); i ? cc.lineTo(px(e), py(a)) : cc.moveTo(px(e), py(a)); }
      cc.stroke(); cc.globalAlpha = 1;

    });
    cc.strokeStyle = ink; cc.lineWidth = 1; cc.setLineDash([4, 4]); cc.beginPath(); cc.moveTo(px(st.eps), m.t); cc.lineTo(px(st.eps), m.t + ph); cc.stroke(); cc.setLineDash([]);
  }

  function redraw() {
    const st = read(), S = st.model === 'sat' ? 200 : 1;
    $('epsOut').textContent = st.eps.toFixed(2);
    drawPlane(st, S); drawCurve(st, S);
    const clean = accuracy('noise', 0, S), acc = accuracy(st.attack, st.eps, S), adp = accuracy('adaptive', st.eps, S);
    const pct = v => Math.round(v * 100) + '%';
    let msg = `<b>${NAMES[st.attack]}</b> at ε = ${st.eps.toFixed(2)}: accuracy <b>${pct(acc)}</b> (clean ${pct(clean)}).`;
    if (st.model === 'sat' && st.attack === 'naive' && st.eps > .3 && adp < acc - .15) msg += ` <span class="ok">The model looks robust.</span> <span class="bad">Switch to the adaptive attack and it falls to ${pct(adp)}.</span> The defence only hid the gradient.`;
    else if (st.attack === 'adaptive' && st.model === 'sat' && st.eps > .3) msg += ` <span class="bad">The adaptive attack ignores the masked gradient and uses the logit directly.</span>`;
    else if (st.model === 'std' && st.attack !== 'noise' && st.eps > .3) msg += ` On the standard model the naive and adaptive attacks agree, so the weakness is visible.`;
    $('readout').innerHTML = msg;
  }
  document.querySelectorAll('#labRoot input').forEach(i => i.addEventListener('input', redraw));
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(redraw);
  redraw();
})();
