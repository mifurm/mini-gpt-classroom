import { createHash } from "node:crypto";
import { copyFile, mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../", import.meta.url));
export const outputDirectory = join(root, ".site");
const load = async path => JSON.parse(await readFile(join(root, path), "utf8"));

export async function buildSite(destination = outputDirectory) {
  const microscope = await load("artifacts/microscope.json");
  for (const [path, expected] of [
    ["artifacts/final.pt", microscope.provenance.checkpoint_sha256],
    ...["model.py", "build_microscope.py"].map(path => [path, microscope.provenance.source_hashes[path]]),
  ]) {
    const actual = createHash("sha256").update(await readFile(join(root, path))).digest("hex");
    if (actual !== expected) throw new Error(`Stale microscope recording for ${path}; run python3 build_microscope.py`);
  }
  const [original, expanded, versions, fivefold, concepts] = await Promise.all([
    load("artifacts/results.json"),
    load("artifacts/versions/expanded_comparison.json"),
    load("artifacts/versions/comparison.json"),
    load("artifacts/fivefold/comparison.json"),
    load("concepts/results.json"),
  ]);
  const summary = {
    original: {
      selected: original.selected,
      test: original.final_test,
      unigram: original.unigram_test,
      rounds: original.rounds.map(record => ({
        name: record.name, change: record.change, validation: record.validation,
        samples: record.samples, curve: record.curve.map(event => ({
          step: event.step, loss: event.validation.loss,
        })),
      })),
    },
    versions: [...expanded.baselines, ...expanded.versions].map(record => ({
      name: record.name, parameters: record.parameters, config: record.config,
      original: record.tests.per_book.original,
      newBooks: record.tests.new_books_macro,
      samples: record.samples ?? versions.versions.find(v => v.name === record.name).samples,
    })),
    fivefold: fivefold.records.filter(record => /^v[1-7](_long)?$/.test(record.name)).map(record => ({
      name: record.name, arm: record.arm, bytes: record.available_train_bytes,
      growth: record.corpus_expansion_factor, steps: record.steps,
      original: record.tests.original, additional: record.tests.additional_macro,
      samples: record.samples,
    })),
    concepts: concepts.models.map(record => ({
      name: record.name, parameters: record.parameters, test: record.test,
      ...(record.longer_test ? { longerTest: record.longer_test } : {}),
    })),
  };
  await mkdir(destination, { recursive: true });
  const copies = [
    ["web/index.html", "index.html"], ["web/style.css", "style.css"],
    ["web/app.js", "app.js"], ["web/404.html", "404.html"],
    ["web/microscope.js", "microscope.js"], ["artifacts/microscope.json", "microscope.json"],
    ["web/staticwebapp.config.json", "staticwebapp.config.json"],
    ...["slides.html", "slides.pdf", "deep-dive.html", "deep-dive.pdf", "deep-dive.pptx",
      "concepts/lesson.html", "concepts/lesson.pdf", "concepts/lesson.pptx",
      "RESULTS.md", "VERSIONS.md", "EXPANDED_RESULTS.md", "FIVEFOLD_RESULTS.md",
      "CONCEPT_MODELS.md", "concepts/RESULTS.md", "artifacts/fivefold/metrics.csv"]
      .map(path => [path, path]),
  ];
  await Promise.all(copies.map(async ([source, target]) => {
    await mkdir(dirname(join(destination, target)), { recursive: true });
    await copyFile(join(root, source), join(destination, target));
  }));
  await writeFile(join(destination, "results.json"), JSON.stringify(summary, null, 2) + "\n");
  const paths = [...copies.map(([, target]) => target), "results.json"].sort();
  const actual = (await readdir(destination, { recursive: true, withFileTypes: true }))
    .filter(entry => entry.isFile())
    .map(entry => relative(destination, join(entry.parentPath, entry.name)));
  const unexpected = actual.filter(path => path !== "manifest.json" && !paths.includes(path));
  if (unexpected.length) throw new Error(`Refusing to publish unexpected files: ${unexpected.join(", ")}`);
  const manifest = await Promise.all(paths.filter(path => path !== "staticwebapp.config.json").map(async path => {
    const body = await readFile(join(destination, path));
    return { path, bytes: body.length, sha256: createHash("sha256").update(body).digest("hex") };
  }));
  await writeFile(join(destination, "manifest.json"), JSON.stringify({ files: manifest }, null, 2) + "\n");
  return { summary, manifest };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const { manifest } = await buildSite();
  console.log(`Built ${manifest.length + 1} public files and hosting configuration in ${outputDirectory}`);
}
