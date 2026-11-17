import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { writeFile, readFile, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL(".", import.meta.url));
const prefix = process.argv[2];
const files = prefix ? {
  specification: `${prefix}.json`, html: `${prefix}.html`, pdf: `${prefix}.pdf`,
  preview: `${prefix}-preview.png`, verification: `${prefix}-render-check.json`,
} : {
  specification: "deep_dive_slides.json", html: "deep-dive.html", pdf: "deep-dive.pdf",
  preview: "deep-dive-preview.png", verification: "deep_dive_render_check.json",
};
const deck = JSON.parse(await readFile(join(root, files.specification), "utf8"));
const executable = process.env.SLIDE_BROWSER || [
  "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  "/usr/bin/chromium", "/usr/bin/google-chrome",
].find(path => existsSync(path));
if (!executable) throw new Error("Set SLIDE_BROWSER to a Chromium/Chrome/Edge executable.");
const profile = await mkdtemp(join(tmpdir(), "mini-gpt-deep-deck-"));
const browser = spawn(executable, [
  "--headless", "--no-first-run", "--no-default-browser-check",
  "--remote-debugging-address=127.0.0.1", "--remote-debugging-port=0",
  `--user-data-dir=${profile}`, "about:blank",
], { stdio: ["ignore", "ignore", "pipe"] });
let socket;
try {
  const url = await new Promise((resolve, reject) => {
    let output = "";
    const timer = setTimeout(() => reject(new Error("Browser startup timed out")), 30000);
    browser.on("error", error => { clearTimeout(timer); reject(error); });
    browser.stderr.on("data", chunk => {
      output += chunk;
      const match = output.match(/DevTools listening on (ws:\/\/[^\s]+)/);
      if (match) { clearTimeout(timer); resolve(match[1]); }
    });
  });
  socket = new WebSocket(url);
  await new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject; });
  let id = 0;
  const pending = new Map();
  socket.onmessage = event => {
    const message = JSON.parse(event.data);
    const callback = pending.get(message.id);
    if (!callback) return;
    pending.delete(message.id);
    clearTimeout(callback.timer);
    if (message.error) callback.reject(new Error(JSON.stringify(message.error)));
    else callback.resolve(message.result);
  };
  const send = (method, params = {}, sessionId) => new Promise((resolve, reject) => {
    const requestId = ++id;
    const timer = setTimeout(() => {
      pending.delete(requestId);
      reject(new Error(`Browser request timed out: ${method}`));
    }, 90000);
    pending.set(requestId, { resolve, reject, timer });
    socket.send(JSON.stringify({ id: requestId, method, params, sessionId }));
  });
  const { targetId } = await send("Target.createTarget", { url: new URL(files.html, `file://${root}`).href });
  const { sessionId } = await send("Target.attachToTarget", { targetId, flatten: true });
  const page = (method, params = {}) => send(method, params, sessionId);
  const evaluate = async expression => {
    const response = await page("Runtime.evaluate", { expression, returnByValue: true });
    if (response.exceptionDetails || response.result.subtype === "error") {
      throw new Error(response.result.description || JSON.stringify(response.exceptionDetails));
    }
    return response.result.value;
  };
  await page("Page.enable");
  await page("Emulation.setDeviceMetricsOverride", {
    width: 1440, height: 950, deviceScaleFactor: 1, mobile: false,
  });
  for (let attempt = 0; attempt < 50; attempt++) {
    if (await evaluate("document.readyState==='complete' && typeof show==='function'")) break;
    if (attempt === 49) throw new Error("Deck failed to load.");
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  const result = await evaluate(`(() => {
    const assert=(condition,message)=>{if(!condition)throw Error(message)};
    assert(slides.length===${deck.slide_count},"Slide count");
    assert(document.querySelectorAll(".notes").length===slides.length,"Speaker notes");
    const ids=[...document.querySelectorAll("[id]")].map(e=>e.id);
    assert(new Set(ids).size===ids.length,"Duplicate HTML ids");
    document.querySelector("#next").click();assert(index===1,"Next button");
    document.dispatchEvent(new KeyboardEvent("keydown",{key:"ArrowLeft"}));assert(index===0,"Keyboard");
    document.querySelector("#notes").click();
    assert(document.body.classList.contains("show-notes"),"Notes toggle");
    document.querySelector("#notes").click();
    const destination=Math.min(12,slides.length-1);
    document.querySelector("#jump").value=destination;
    document.querySelector("#jump").dispatchEvent(new Event("change"));assert(index===destination,"Slide picker");
    const overflows=[];
    for(let i=0;i<slides.length;i++){
      show(i);
      const s=slides[i],content=s.querySelector(".content"),footer=s.querySelector("footer");
      const bounds=content.getBoundingClientRect(),bottom=footer.getBoundingClientRect().top;
      if(s.scrollWidth>s.clientWidth+1 || bounds.bottom>bottom-6){
        overflows.push({slide:i+1,title:s.querySelector("h1").textContent,bottom:bounds.bottom,limit:bottom-6});
      }
    }
    assert(overflows.length===0,"Screen overflow: "+JSON.stringify(overflows));
    show(0);
    return {slides:slides.length,navigation:"passed",notes:"passed",screen_layout:"passed"};
  })()`);
  console.log(JSON.stringify(result));
  await page("Emulation.setEmulatedMedia", { media: "print" });
  const overflows = await evaluate(`slides.map((s,i)=>{
    const content=s.querySelector(".content").getBoundingClientRect();
    const footer=s.querySelector("footer").getBoundingClientRect();
    return {slide:i+1,bottom:content.bottom,limit:footer.top-6};
  }).filter(x=>x.bottom>x.limit)`);
  if (overflows.length) throw new Error(`Print overflow: ${JSON.stringify(overflows)}`);
  const pdf = await page("Page.printToPDF", {
    preferCSSPageSize: true, printBackground: true, displayHeaderFooter: false,
  });
  await writeFile(join(root, files.pdf), Buffer.from(pdf.data, "base64"));
  await page("Emulation.setEmulatedMedia", { media: "" });
  await evaluate("show(Math.min(9,slides.length-1))");
  const screenshot = await page("Page.captureScreenshot", {
    format: "png", clip: { x: 80, y: 24, width: 1280, height: 720, scale: 1 },
  });
  await writeFile(join(root, files.preview), Buffer.from(screenshot.data, "base64"));
  await writeFile(join(root, files.verification), JSON.stringify({
    ...result, print_layout: "passed", pdf: files.pdf, screenshot: files.preview,
  }, null, 2) + "\n");
  console.log(`Rendered ${files.pdf} (${deck.slide_count} slides) and a preview.`);
  await send("Browser.close");
} catch (error) {
  console.error("Deck render validation failed:", error);
  throw error;
} finally {
  socket?.close();
  if (browser.exitCode === null && browser.signalCode === null) {
    await Promise.race([
      new Promise(resolve => browser.once("exit", resolve)),
      new Promise(resolve => setTimeout(resolve, 2000)),
    ]);
    if (browser.exitCode === null && browser.signalCode === null) {
      const stopped = new Promise(resolve => browser.once("exit", resolve));
      browser.kill("SIGTERM");
      await stopped;
    }
  }
  await rm(profile, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
}
