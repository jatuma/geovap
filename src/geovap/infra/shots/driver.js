#!/usr/bin/env node
/**
 * Headless Chromium screenshot driver for pointcloud-tools/consolidated/index.html
 * (plan Part C6). Takes a JSON job list, groups shots by page load so octrees are
 * fetched once, and drives window.__ready / window.__setCam / window.__setOpacity
 * exposed by consolidated/index.html.
 *
 * Usage:
 *   node driver.js --jobs jobs.json [--out-dir DIR] [--width 1600] [--height 900]
 *   node driver.js --help
 *
 * jobs.json shape:
 *   [
 *     {
 *       "url": "http://localhost:8080/pointclouds/consolidated/index.html?mode=rgb&panos=panos_corr0&frame=12",
 *       "shots": [
 *         {"out": "f0012_y0_op0.png",   "yaw": 0,   "pitch": 0, "opacity": 0},
 *         {"out": "f0012_y0_op1.png",   "yaw": 0,   "pitch": 0, "opacity": 1},
 *         {"out": "f0012_y90_op0.5.png","yaw": 1.5707963, "opacity": 0.5}
 *       ]
 *     },
 *     ...
 *   ]
 *
 * Falls back path (no playwright installed): use `chrome --headless=new --screenshot`
 * per shot instead (see screenshots.py, which drives this or that fallback).
 */
"use strict";

const fs = require("fs");
const path = require("path");

function parseArgs(argv) {
  const args = { outDir: ".", width: 1600, height: 900 };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--help" || a === "-h") { args.help = true; }
    else if (a === "--jobs") { args.jobs = argv[++i]; }
    else if (a === "--out-dir") { args.outDir = argv[++i]; }
    else if (a === "--width") { args.width = parseInt(argv[++i], 10); }
    else if (a === "--height") { args.height = parseInt(argv[++i], 10); }
    else if (a === "--timeout-ms") { args.timeoutMs = parseInt(argv[++i], 10); }
  }
  return args;
}

function printHelp() {
  console.log(`node driver.js --jobs <jobs.json> [--out-dir DIR] [--width 1600] [--height 900] [--timeout-ms 30000]

Drives headless Chromium (via playwright) against consolidated/index.html: for each
job.url, waits for window.__ready, then for each shot calls window.__setCam(yaw, pitch,
fov) / window.__setOpacity(opacity) (only the given fields), waits for
Potree.numNodesLoading === 0, and screenshots to <out-dir>/<shot.out>.

CHROME env var (or ~/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome) selects
the Chromium binary; launched with --use-angle=swiftshader --enable-unsafe-swiftshader
for headless WebGL2.`);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help || !args.jobs) {
    printHelp();
    process.exit(args.help ? 0 : 1);
    return;
  }

  const { chromium } = require("playwright");

  const jobs = JSON.parse(fs.readFileSync(args.jobs, "utf8"));
  const timeoutMs = args.timeoutMs || 180000;  // first paint of a 25 GB octree under swiftshader is slow

  const home = process.env.HOME || "";
  const defaultChrome = path.join(home, ".cache/ms-playwright/chromium-1234/chrome-linux64/chrome");
  const executablePath = process.env.CHROME || (fs.existsSync(defaultChrome) ? defaultChrome : undefined);

  const browser = await chromium.launch({
    executablePath,
    args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
  });

  const results = [];
  try {
    const page = await browser.newPage({ viewport: { width: args.width, height: args.height } });
    page.on("console", (msg) => { if (msg.type() === "error") console.error("[page]", msg.text()); });

    for (const job of jobs) {
      const t0 = Date.now();
      // Potree.numNodesLoading occasionally never returns to 0 under swiftshader (a hung node
      // request), so a page that does not become ready in time is reloaded once before giving up.
      let ready = false;
      for (let attempt = 0; attempt < 2 && !ready; attempt++) {
        try {
          await page.goto(job.url, { waitUntil: "load", timeout: timeoutMs });
          await page.waitForFunction("window.__ready === true", null, { timeout: timeoutMs });
          ready = true;
        } catch (e) {
          if (attempt === 1) throw e;
          console.error(`[driver] not ready after ${timeoutMs} ms, reloading once: ${job.url}`);
        }
      }

      for (const shot of job.shots || []) {
        await page.evaluate(([yaw, pitch, fov, opacity]) => {
          if ((yaw !== null || pitch !== null || fov !== null) && window.__setCam) {
            window.__setCam(yaw, pitch, fov);
          }
          if (opacity !== null && window.__setOpacity) {
            window.__setOpacity(opacity);
          }
        }, [shot.yaw ?? null, shot.pitch ?? null, shot.fov ?? null, shot.opacity ?? null]);

        // settle: wait until node streaming pauses (LOD refinement never fully stops on a big octree,
        // so give up after 20 s and shoot what we have), then a few frames for the material change
        try {
          await page.waitForFunction("typeof Potree === 'undefined' || Potree.numNodesLoading === 0", null, { timeout: 20000 });
        } catch (e) { /* still streaming: acceptable */ }
        await page.waitForTimeout(600);

        const outPath = path.join(args.outDir, shot.out);
        fs.mkdirSync(path.dirname(outPath), { recursive: true });
        await page.screenshot({ path: outPath });
        results.push({ url: job.url, out: shot.out, ok: true });
      }
      console.error(`[driver] ${job.url} -> ${(job.shots || []).length} shots in ${Date.now() - t0} ms`);
    }
  } catch (err) {
    console.error("[driver] error:", err.stack || err);
    results.push({ ok: false, error: String(err) });
  } finally {
    await browser.close();
  }

  console.log(JSON.stringify(results));
  process.exit(results.some((r) => r.ok === false) ? 1 : 0);
}

main();
