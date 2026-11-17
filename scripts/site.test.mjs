import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { buildSite } from "./build-site.mjs";

test("website projects recorded metrics and samples without publishing models or corpora", async () => {
  const directory = await mkdtemp(join(tmpdir(), "mini-gpt-site-"));
  try {
    const { summary, manifest } = await buildSite(directory);
    const source = JSON.parse(await readFile(new URL("../artifacts/results.json", import.meta.url)));
    assert.deepEqual(summary.original.test, source.final_test);
    assert.deepEqual(summary.original.rounds[0].samples, source.rounds[0].samples);
    assert.equal(summary.versions.length, 7);
    assert.equal(summary.fivefold.length, 15);
    const fivefold = JSON.parse(await readFile(new URL("../artifacts/fivefold/comparison.json", import.meta.url)));
    for (const record of summary.fivefold) {
      const original = fivefold.records.find(item => item.name === record.name && item.arm === record.arm);
      assert.deepEqual(record.additional, original.tests.additional_macro);
      assert.equal(record.bytes, original.available_train_bytes);
    }
    assert.equal(summary.concepts.length, 3);
    const microscope = JSON.parse(await readFile(join(directory, "microscope.json")));
    const recorded = JSON.parse(await readFile(new URL("../artifacts/microscope.json", import.meta.url)));
    assert.deepEqual(microscope, recorded);
    assert.equal(microscope.training.updates, 1);
    assert.equal(microscope.generation.steps.length, microscope.generation.count);
    assert.ok(manifest.some(file => file.path === "microscope.js"));
    assert.ok(manifest.every(file => !/\.(pt|bin|tfstate|py)$/.test(file.path)));
    const html = await readFile(join(directory, "index.html"), "utf8");
    const targets = [...html.matchAll(/(?:href|src)="([^"#]+)"/g)].map(match => match[1]);
    for (const target of targets.filter(target => !target.startsWith("https://"))) {
      assert.ok(manifest.some(file => file.path === target), `Missing linked file: ${target}`);
    }
    await writeFile(join(directory, "secret.txt"), "not publishable");
    await assert.rejects(buildSite(directory), /Refusing to publish unexpected files: secret.txt/);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
