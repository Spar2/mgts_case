async (page) => {
  await page.locator(".heat-figure canvas").waitFor();
  await page.evaluate(async () => {
    const state = {
      startedAt: new Date().toISOString(),
      started: performance.now(),
      durationMs: 1800000,
      done: false,
      userAgent: navigator.userAgent,
      viewport: [innerWidth, innerHeight],
      devicePixelRatio,
      buildAssets: [...document.querySelectorAll('script[src]')].map(s=>new URL(s.src).pathname),
      origin: location.origin,
      frames: [],
      frameCount: 0,
      longTasks: 0,
      longTaskMaxMs: 0,
      samples: [],
      replayErrors: 0,
      runtimeErrors: [],
    };
    window.__dtsSoak = state;
    window.addEventListener('error',event=>{if(state.runtimeErrors.length<100)state.runtimeErrors.push(event.message);});
    const observer = new PerformanceObserver((list) => {
      for (const e of list.getEntries()) {
        state.longTasks++;
        state.longTaskMaxMs = Math.max(state.longTaskMaxMs, e.duration);
      }
    });
    observer.observe({ type: "longtask", buffered: false });
    let last = performance.now();
    function frame(now) {
      if (state.done) return;
      state.frames[state.frameCount % 3600] = now - last;
      state.frameCount++;
      last = now;
      if (!state.done) requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
    const sample = () =>
      state.samples.push({
        elapsedS: Math.round((performance.now() - state.started) / 1000),
        heapMiB: performance.memory?.usedJSHeapSize / 1048576,
        cacheMiB: window.__dtsCacheStats().bytes / 1048576,
        canvases: document.querySelectorAll("canvas").length,
        domNodes: document.querySelectorAll("*").length,
        visibility: document.visibilityState,
        row: document.querySelector('#row')?.value,
        graphics: window.__dtsGraphicsStats?.(),
      });
    sample();
    const sampling = setInterval(sample, 15000);
    let loopBusy = false;
    const loop = setInterval(async () => {
      if (loopBusy) return;
      const button = [...document.querySelectorAll("button")].find(
        (b) => b.textContent === "Воспроизвести",
      );
      if (button && !button.disabled) {
        loopBusy = true;
        try {
          const id = document.querySelector(".replay-bar").dataset.replayId;
          const token = (await (await fetch("/api/v1/session")).json())
            .mutation_token;
          for (const body of [{ action: "seek", row: 0 }, { action: "play" }])
            { const response = await fetch(`/api/v1/replays/${id}/commands`, {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
                "X-Workbench-Token": token,
              },
              body: JSON.stringify(body),
            }); if(!response.ok)state.replayErrors++; }
        } finally {
          loopBusy = false;
        }
      }
    }, 3000);
    setTimeout(() => {
      state.done = true;
      state.endedAt = new Date().toISOString();
      sample();
      clearInterval(sampling);
      clearInterval(loop);
      observer.disconnect();
      const sorted = state.frames.slice().sort((a, b) => a - b);
      state.frameP95Ms = sorted[Math.floor(sorted.length * 0.95)];
      delete state.frames;
    }, state.durationMs);
    return { startedAt: state.startedAt, durationMs: state.durationMs };
  });
  await page
    .getByRole("combobox", { name: "Скорость replay" })
    .selectOption("60");
  await page
    .getByRole("button", { name: "Воспроизвести", exact: true })
    .click();
}
