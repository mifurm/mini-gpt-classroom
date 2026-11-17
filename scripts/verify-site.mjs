import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { outputDirectory } from "./build-site.mjs";

const base = new URL(process.argv[2]);
assert.equal(base.protocol, "https:", "Verify the public HTTPS endpoint");
const manifest = JSON.parse(await readFile(join(outputDirectory, "manifest.json"), "utf8"));
for (const file of [...manifest.files, { path: "manifest.json" }]) {
  const response = await fetch(new URL(file.path, base), { signal: AbortSignal.timeout(30000) });
  assert.equal(response.status, 200, `${file.path}: HTTP status`);
  const body = Buffer.from(await response.arrayBuffer());
  const expected = await readFile(join(outputDirectory, file.path));
  assert.equal(createHash("sha256").update(body).digest("hex"),
    createHash("sha256").update(expected).digest("hex"), `${file.path}: deployed bytes differ`);
  const types = { html: "text/html", css: "text/css", js: "javascript", json: "application/json", pdf: "application/pdf" };
  const type = types[file.path.split(".").at(-1)];
  if (type) assert.ok(response.headers.get("content-type")?.includes(type), `${file.path}: MIME type`);
}
const missing = await fetch(new URL("__missing_lab_page__", base), { signal: AbortSignal.timeout(30000) });
assert.equal(missing.status, 404, "Unknown paths must return HTTP 404");
assert.match(await missing.text(), /Page not found/);
console.log(`Verified ${manifest.files.length + 1} public files byte-for-byte, MIME types, and custom 404 at ${base}`);
