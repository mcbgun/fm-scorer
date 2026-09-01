/* Shared front-end behaviour: DataTables init, player drawer, drag-and-drop uploads. */
(function () {
  function initTables() {
    if (!window.jQuery || !jQuery.fn.dataTable) return;
    jQuery("table.js-datatable").each(function () {
      var $t = jQuery(this);
      if (jQuery.fn.dataTable.isDataTable(this)) return;
      var order = $t.data("order-col");
      var dir = $t.data("order-dir") || "desc";
      $t.DataTable({
        pageLength: parseInt($t.data("page-length") || "25", 10),
        lengthMenu: [10, 25, 50, 100, 250],
        order: order !== undefined ? [[parseInt(order, 10), dir]] : [],
        scrollX: true,
        fixedHeader: false,
        stateSave: true,
        language: { search: "Filter:" },
      });
    });
  }

  function openPlayer(source, idx, name) {
    var el = document.getElementById("playerDrawer");
    if (!el) return;
    var body = document.getElementById("playerDrawerBody");
    var title = document.getElementById("playerDrawerTitle");
    var link = document.getElementById("playerDrawerLink");
    title.textContent = name || "Player";
    link.href = "/player/" + source + "/" + idx;
    body.innerHTML = '<div class="text-muted p-3">Loading…</div>';
    bootstrap.Offcanvas.getOrCreateInstance(el).show();
    fetch("/player/" + source + "/" + idx + "?partial=1")
      .then(function (r) { return r.text(); })
      .then(function (html) { body.innerHTML = html; drawRadars(body); drawProjections(body); })
      .catch(function () { body.innerHTML = '<div class="alert alert-danger">Could not load player.</div>'; });
  }

  document.addEventListener("click", function (e) {
    var a = e.target.closest("[data-player-source]");
    if (!a) return;
    e.preventDefault();
    openPlayer(a.dataset.playerSource, a.dataset.playerIdx, a.dataset.playerName);
  });

  /* Radar chart on a <canvas data-radar='{"labels":[],"values":[],"lo":[],"hi":[]}'> */
  function drawRadars(root) {
    (root || document).querySelectorAll("canvas[data-radar]").forEach(function (c) {
      var d = JSON.parse(c.dataset.radar);
      var ctx = c.getContext("2d");
      var w = c.width, h = c.height, cx = w / 2, cy = h / 2, R = Math.min(w, h) / 2 - 28;
      var n = d.labels.length;
      if (!n) return;
      ctx.clearRect(0, 0, w, h);
      ctx.strokeStyle = "#dee2e6";
      [5, 10, 15, 20].forEach(function (lvl) {
        ctx.beginPath();
        for (var i = 0; i < n; i++) {
          var ang = -Math.PI / 2 + (2 * Math.PI * i) / n, r = (R * lvl) / 20;
          var x = cx + r * Math.cos(ang), y = cy + r * Math.sin(ang);
          i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
        }
        ctx.closePath(); ctx.stroke();
      });
      function poly(vals, fill, stroke) {
        ctx.beginPath();
        for (var i = 0; i < n; i++) {
          var ang = -Math.PI / 2 + (2 * Math.PI * i) / n, r = (R * Math.max(0, Math.min(20, vals[i]))) / 20;
          var x = cx + r * Math.cos(ang), y = cy + r * Math.sin(ang);
          i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
        }
        ctx.closePath();
        ctx.fillStyle = fill; ctx.fill();
        ctx.strokeStyle = stroke; ctx.lineWidth = 1.5; ctx.stroke();
      }
      if (d.hi && d.lo) poly(d.hi, "rgba(255,193,7,0.15)", "rgba(255,193,7,0.6)");
      poly(d.values, "rgba(13,110,253,0.25)", "#0d6efd");
      ctx.fillStyle = "#212529"; ctx.font = "11px sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      for (var i = 0; i < n; i++) {
        var ang = -Math.PI / 2 + (2 * Math.PI * i) / n;
        var x = cx + (R + 16) * Math.cos(ang), y = cy + (R + 16) * Math.sin(ang);
        ctx.fillText(d.labels[i] + " " + d.values[i], x, y);
      }
    });
  }

  /* Projection sparkline on <canvas data-projection='{"mid":[],"lo":[],"hi":[]}'> */
  function drawProjections(root) {
    (root || document).querySelectorAll("canvas[data-projection]").forEach(function (c) {
      var d = JSON.parse(c.dataset.projection);
      var ctx = c.getContext("2d");
      var w = c.width, h = c.height, pad = 26;
      var all = d.lo.concat(d.hi);
      var lo = Math.floor(Math.min.apply(null, all)) - 0.5, hi = Math.ceil(Math.max.apply(null, all)) + 0.5;
      var n = d.mid.length;
      function X(i) { return pad + ((w - 2 * pad) * i) / (n - 1); }
      function Y(v) { return h - pad - ((h - 2 * pad) * (v - lo)) / (hi - lo); }
      ctx.clearRect(0, 0, w, h);
      ctx.beginPath();
      for (var i = 0; i < n; i++) ctx.lineTo(X(i), Y(d.hi[i]));
      for (var j = n - 1; j >= 0; j--) ctx.lineTo(X(j), Y(d.lo[j]));
      ctx.closePath(); ctx.fillStyle = "rgba(13,110,253,0.12)"; ctx.fill();
      ctx.beginPath(); ctx.strokeStyle = "#0d6efd"; ctx.lineWidth = 2;
      for (var k = 0; k < n; k++) k ? ctx.lineTo(X(k), Y(d.mid[k])) : ctx.moveTo(X(k), Y(d.mid[k]));
      ctx.stroke();
      ctx.fillStyle = "#6c757d"; ctx.font = "10px sans-serif"; ctx.textAlign = "center";
      for (var s = 0; s < n; s++) ctx.fillText(s === 0 ? "now" : "+" + s, X(s), h - 8);
      ctx.textAlign = "left";
      ctx.fillText(hi.toFixed(0), 2, pad); ctx.fillText(lo.toFixed(0), 2, h - pad);
    });
  }

  function initDropzones() {
    document.querySelectorAll(".dropzone").forEach(function (dz) {
      var input = dz.querySelector("input[type=file]");
      if (!input) return;
      ["dragenter", "dragover"].forEach(function (ev) { dz.addEventListener(ev, function (e) { e.preventDefault(); dz.classList.add("dragover"); }); });
      ["dragleave", "drop"].forEach(function (ev) { dz.addEventListener(ev, function (e) { e.preventDefault(); dz.classList.remove("dragover"); }); });
      dz.addEventListener("drop", function (e) { input.files = e.dataTransfer.files; input.dispatchEvent(new Event("change")); });
      input.addEventListener("change", function () {
        var lbl = dz.querySelector(".dropzone-file");
        if (lbl) lbl.textContent = input.files.length ? input.files[0].name : "";
      });
    });
    document.querySelectorAll("form[data-busy]").forEach(function (f) {
      f.addEventListener("submit", function () {
        var btn = f.querySelector("button[type=submit]");
        if (btn) { btn.disabled = true; btn.textContent = f.dataset.busy; }
      });
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    initTables(); drawRadars(); drawProjections(); initDropzones();
  });
  window.fmOpenPlayer = openPlayer;
  window.fmDrawCharts = function (root) { drawRadars(root); drawProjections(root); };
})();
