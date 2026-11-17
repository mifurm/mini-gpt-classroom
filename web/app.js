const percent = value => `${(value * 100).toFixed(2)}%`;
const number = value => value.toLocaleString("en-US");
const element = (tag, text, className) => {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
};
const row = (target, values) => {
  const tr = element("tr");
  values.forEach(value => tr.append(element("td", String(value))));
  document.getElementById(target).append(tr);
};
const card = (target, title, value, detail) => {
  const article = element("article");
  article.append(element("h3", title), element("span", value, "value"), element("p", detail, "detail"));
  document.getElementById(target).append(article);
};

async function main() {
  const response = await fetch("results.json");
  if (!response.ok) throw new Error(`Results request failed: HTTP ${response.status}`);
  const data = await response.json();
  card("metrics", "Final test accuracy", percent(data.original.test.accuracy),
    `${number(data.original.test.targets)} heldout next-byte targets`);
  card("metrics", "Final bits per byte", data.original.test.bits_per_byte.toFixed(3),
    `Unigram baseline: ${data.original.unigram.bits_per_byte.toFixed(3)} bits/byte (lower is better)`);
  card("metrics", "Selected recipe", data.original.selected.replaceAll("_", " "),
    "Selected by validation loss, not by the test set");
  data.original.rounds.forEach(record => row("rounds", [
    record.name, record.change, record.validation.loss.toFixed(3), percent(record.validation.accuracy),
  ]));
  data.versions.forEach(record => row("versions", [
    record.name.toUpperCase(), number(record.parameters),
    `${record.config.layers} / ${record.config.context} bytes`,
    percent(record.original.accuracy), percent(record.newBooks.accuracy),
  ]));
  data.fivefold.filter(record => record.arm === "large" && record.name !== "v7_long").forEach(record => {
    const control = data.fivefold.find(item => item.name === record.name && item.arm === "control");
    const difference = 100 * (record.additional.accuracy - control.additional.accuracy);
    row("fivefold-rows", [record.name.toUpperCase(), number(record.bytes), `${record.growth.toFixed(2)}x`,
      percent(control.additional.accuracy), percent(record.additional.accuracy),
      `${difference >= 0 ? "+" : ""}${difference.toFixed(2)} pp`]);
  });
  const long = data.fivefold.find(record => record.name === "v7_long");
  document.getElementById("long-run").textContent =
    `With a longer ${number(long.steps)}-update budget, V7 reaches ${percent(long.additional.accuracy)} on the additional books. This is not a matched-budget comparison.`;
  const titles = { D1: "Supervised decision policy", R0: "Answer-only control", R1: "Scratchpad reasoning" };
  data.concepts.forEach(record => card("concept-cards", `${record.name}: ${titles[record.name]}`,
    percent(record.test.accuracy ?? record.test.answer_correct),
    `${number(record.parameters)} parameters; ${record.test.cases} heldout cases. ${record.name === "D1" ? `Mean regret: ${record.test.mean_regret}.` : "Final-answer accuracy."}`));
  const sampleModels = [
    ...data.original.rounds.map(record => ({ ...record, label: `Original / ${record.name}` })),
    ...data.versions.map(record => ({ ...record, label: `Version / ${record.name.toUpperCase()}` })),
  ];
  const select = document.getElementById("sample-model");
  sampleModels.forEach((record, index) => {
    const option = element("option", record.label);
    option.value = String(index);
    select.append(option);
  });
  const renderSamples = () => {
    const output = document.getElementById("sample-output");
    output.replaceChildren();
    sampleModels[Number(select.value)].samples.forEach(sample => {
      const article = element("article");
      article.append(element("h3", `Prompt: ${sample.prompt}`),
        element("p", `Seed ${sample.seed} / saved continuation`, "detail"),
        element("pre", sample.continuation));
      output.append(article);
    });
  };
  select.addEventListener("change", renderSamples);
  renderSamples();
  document.getElementById("status").textContent = "";
}

main().catch(error => {
  console.error("Lab results could not be displayed", error);
  const status = document.getElementById("status");
  status.setAttribute("role", "alert");
  status.textContent = `Unable to load the results explorer: ${error.message}. Teaching materials and report downloads remain available below.`;
});
