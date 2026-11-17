const byId = id => document.getElementById(id);
const make = (tag, text) => {
  const result = document.createElement(tag);
  if (text !== undefined) result.textContent = text;
  return result;
};
const byte = id => id === 32 ? "space" : id === 10 ? "\\n" :
  id >= 33 && id <= 126 ? String.fromCharCode(id) : `0x${id.toString(16).padStart(2, "0").toUpperCase()}`;
const fixed = value => value.toFixed(5);
const percent = value => `${(100 * value).toFixed(2)}%`;
const vector = values => `[${values.map(fixed).join(", ")}]`;

function table(target, caption, headings, rows) {
  const wrapper = make("div");
  wrapper.className = "table-wrap";
  const grid = make("table");
  grid.append(make("caption", caption));
  const header = make("tr");
  headings.forEach(heading => header.append(make("th", heading)));
  const head = make("thead");
  head.append(header);
  grid.append(head);
  const body = make("tbody");
  rows.forEach(values => {
    const row = make("tr");
    values.forEach(value => row.append(make("td", String(value))));
    body.append(row);
  });
  grid.append(body);
  wrapper.append(grid);
  target.append(wrapper);
}

function distribution(target, probabilities, logits) {
  target.replaceChildren();
  table(target, "Five highest recorded probabilities",
    logits ? ["Byte", "ID", "Logit", "Probability"] : ["Byte", "ID", "Sampling probability"],
    probabilities.map(item => [
      byte(item.id), item.id, ...(logits ? [fixed(logits[item.id])] : []), percent(item.probability),
    ]));
}

async function main() {
  const response = await fetch("microscope.json");
  if (!response.ok) throw new Error(`Trace request failed: HTTP ${response.status}`);
  const data = await response.json();
  const { forward, training, generation, provenance } = data;
  byId("trace-provenance").textContent =
    `Original checkpoint: ${provenance.checkpoint}; ${provenance.parameters.toLocaleString("en-US")} parameters; ` +
    `${provenance.config.layers} blocks, ${provenance.config.heads} heads, width ${provenance.config.width}. Recorded with PyTorch ${provenance.torch_version} on CPU.`;
  byId("tokenizer-example").textContent =
    `This model uses ${data.tokenizer.vocabulary_size} byte IDs, not words or GPT-2 subwords. ` +
    `For example, ${JSON.stringify(data.tokenizer.text)} has ${[...data.tokenizer.text].length} characters but ` +
    `${data.tokenizer.ids.length} UTF-8 bytes: [${data.tokenizer.ids.join(", ")}]. ` +
    `Our recorded prefix is ${JSON.stringify(forward.text)}.`;
  const fill = (id, values) => values.forEach(([value, text]) => {
    const option = make("option", text);
    option.value = value;
    byId(id).append(option);
  });
  fill("prediction-position", forward.ids.map((id, index) => [index, `${index}: ${byte(id)} (ID ${id})`]));
  byId("prediction-position").value = String(forward.ids.length - 1);
  fill("attention-block", forward.blocks.map((_, index) => [index, String(index + 1)]));
  fill("attention-head", Array.from({ length: provenance.config.heads }, (_, index) => [index, String(index + 1)]));
  forward.ids.forEach((id, index) => {
    const item = make("li", `${index}: ${byte(id)} / ${id}`);
    byId("byte-strip").append(item);
  });

  function renderForward() {
    const position = Number(byId("prediction-position").value);
    const blockIndex = Number(byId("attention-block").value);
    const head = Number(byId("attention-head").value);
    const block = forward.blocks[blockIndex];
    const prefix = new TextDecoder().decode(new Uint8Array(forward.ids.slice(0, position + 1)));
    byId("prediction-prefix").textContent =
      `Position ${position} has seen ${JSON.stringify(prefix)} and predicts the byte after it. Later prefix bytes are not visible at this position.`;
    [...byId("byte-strip").children].forEach((item, index) => item.classList.toggle("selected-byte", index === position));
    byId("embedding-shape").textContent = `Token tensor [1, ${forward.ids.length}] becomes hidden states [${forward.combined_embeddings.shape.join(", ")}]: batch, positions, features.`;
    const token = forward.token_embeddings.first_four[0][position];
    const positional = forward.position_embeddings.first_four[position];
    const combined = forward.combined_embeddings.first_four[0][position];
    byId("embedding-values").replaceChildren();
    table(byId("embedding-values"), "Actual values at the selected position",
      ["Feature", "Token embedding", "+ Position embedding", "= Combined"],
      token.map((value, index) => [index, fixed(value), fixed(positional[index]), fixed(combined[index])]));
    byId("attention-values").replaceChildren();
    table(byId("attention-values"), `Block ${blockIndex + 1}, head ${head + 1}, query position ${position}`,
      ["Key position", "Byte", "Attention weight"],
      block.attention[head][position].map((weight, index) =>
        [index, byte(forward.ids[index]), index > position ? "MASKED (0)" : percent(weight)]));
    const check = forward.causality;
    byId("causality-check").textContent =
      `Recorded causality check: changing the final byte (position ${check.changed_position}, ID ${check.original_id} to ${check.changed_id}) ` +
      `changed earlier-position logits by at most ${check.earlier_max_logit_difference}. This check was executed in Python, not recomputed by the selectors.`;
    const incoming = blockIndex === 0 ? forward.combined_embeddings : forward.blocks[blockIndex - 1].output;
    byId("residual-values").replaceChildren();
    table(byId("residual-values"), `Block ${blockIndex + 1}: two residual additions`,
      ["Feature", "Incoming", "+ Attention", "= Residual", "+ MLP", "= Block output"],
      incoming.first_four[0][position].map((value, index) => [
        index, fixed(value), fixed(block.attention_output.first_four[0][position][index]),
        fixed(block.after_attention.first_four[0][position][index]),
        fixed(block.mlp_output.first_four[0][position][index]), fixed(block.output.first_four[0][position][index]),
      ]));
    byId("projection-shape").textContent =
      `After all ${forward.blocks.length} blocks: hidden states [${forward.normalized.shape.join(", ")}] project to logits [${forward.logits_shape.join(", ")}].`;
    byId("normalized-values").textContent =
      `Final normalized state, features 0-3:\n${vector(forward.normalized.first_four[0][position])}`;
    const top = forward.probabilities[position].map((probability, id) => ({ id, probability }))
      .sort((a, b) => b.probability - a.probability).slice(0, 5);
    distribution(byId("prediction-values"), top, forward.logits[position]);
  }
  ["prediction-position", "attention-block", "attention-head"].forEach(id =>
    byId(id).addEventListener("change", renderForward));
  renderForward();

  byId("training-example").textContent =
    `Synthetic teaching text: ${JSON.stringify(training.synthetic_example)}. This update uses only its first ` +
    `${training.input_ids.length + 1} bytes, ${JSON.stringify(training.window)}, as one shifted training window (batch size ${training.batch_size}). ` +
    "The copy starts with the original Archive-trained weights; it is not trained from scratch here.";
  function renderTraining() {
    const stage = Number(byId("training-stage").value);
    const output = byId("training-detail");
    output.replaceChildren();
    if (stage === 0) {
      output.append(make("h4", "1 / Inputs and targets are offset by one byte"),
        make("p", "One causal forward pass predicts the next byte at every position in parallel. Each position sees only its prefix; the target is the byte immediately after it."));
      table(output, "Teacher forcing: the prefix comes from the teaching text",
        ["Position", "Input byte", "Input ID", "Target byte", "Target ID"],
        training.positions.map((item, index) => [index, byte(item.input_id), item.input_id, byte(item.target_id), item.target_id]));
    } else if (stage === 1) {
      output.append(make("h4", "2 / Cross-entropy measures surprise at the correct targets"),
        make("pre", "L = mean over positions of -ln P(correct next byte | prefix)"),
        make("p", `Mean loss before the update: ${fixed(training.loss_before)} nats. A target with higher probability contributes less loss. These are training-window scores, not heldout evaluation.`));
      table(output, "Actual per-position probabilities and losses before learning",
        ["Position", "Correct target", "Probability", "-ln(probability)"],
        training.positions.map((item, index) => [index, byte(item.target_id), percent(item.probability_before), fixed(item.loss_before)]));
    } else if (stage === 2) {
      const weight = training.weight;
      output.append(make("h4", "3 / Backpropagation computes how weights affect loss"),
        make("p", "loss.backward() differentiates the mean loss through the output projection, blocks, and embeddings. Gradients are not probabilities; they describe local sensitivity."),
        make("pre", `Global gradient L2 norm: ${fixed(training.gradient_norm)}\n` +
          `Example parameter: ${weight.name}[${weight.token_id}, ${weight.feature}]\n` +
          `Weight before: ${weight.before.toExponential(8)}\nGradient: ${weight.gradient.toExponential(8)}`),
        make("p", "The example coordinate has the largest absolute gradient in the token embedding matrix. That matrix is also used by the output projection."));
    } else {
      const weight = training.weight;
      const settings = training.optimizer;
      output.append(make("h4", "4 / AdamW applies one real update"),
        make("p", `Learning rate ${settings.learning_rate}; betas ${settings.betas.join(", ")}; epsilon ${settings.epsilon}; weight decay ${settings.weight_decay}. ` +
          "AdamW uses gradient moments plus decoupled weight decay, not simply weight minus learning rate times gradient."),
        make("pre", `${weight.name}[${weight.token_id}, ${weight.feature}]\n` +
          `Before: ${weight.before.toExponential(8)}\nAfter:  ${weight.after.toExponential(8)}\n` +
          `Change: ${(weight.after - weight.before).toExponential(8)}\n` +
          `First moment: ${weight.first_moment.toExponential(8)}\nSecond moment: ${weight.second_moment.toExponential(8)}`),
        make("p", `Re-evaluating the same window: mean loss ${fixed(training.loss_before)} before, ${fixed(training.loss_after)} after. Some individual positions can get worse even when the average improves.`));
      table(output, "Correct-target probability before and after the disposable update",
        ["Position", "Target", "Before", "After"],
        training.positions.map((item, index) => [index, byte(item.target_id), percent(item.probability_before), percent(item.probability_after)]));
    }
  }
  byId("training-stage").addEventListener("change", renderTraining);
  renderTraining();

  byId("generation-settings").textContent =
    `Replaying ${generation.count} next-byte steps from GPT.generate: temperature ${generation.temperature}, top-k ${generation.top_k}, seed ${generation.seed}. ` +
    "Logits are divided by temperature, then values below the top-k cutoff are masked before sampling. The table shows the five highest probabilities after these transformations.";
  byId("generation-step").max = String(generation.count);
  let timer;
  function stopPlayback() {
    clearInterval(timer);
    timer = undefined;
    byId("generation-play").textContent = "Play steps";
  }
  function renderGeneration() {
    const index = Number(byId("generation-step").value) - 1;
    const step = generation.steps[index];
    byId("generation-counter").textContent = ` ${index + 1} / ${generation.count}`;
    byId("generation-prefix").textContent = step.prefix;
    byId("generation-output").textContent = step.text_after;
    byId("generation-context").textContent =
      `This call uses ${step.context_ids.length} byte IDs. The model accepts at most the most recent ${provenance.config.context} bytes; this short example does not need truncation.`;
    distribution(byId("generation-values"), step.candidates);
    byId("generation-sampled").textContent =
      `Sampled byte: ${byte(step.sampled_id)} (ID ${step.sampled_id}), probability ${percent(step.sampled_probability)}. Append it to the prefix for the next call.`;
    byId("generation-prev").disabled = index === 0;
    byId("generation-next").disabled = index === generation.count - 1;
    if (index === generation.count - 1) stopPlayback();
  }
  const move = offset => {
    byId("generation-step").value = String(Math.max(1, Math.min(generation.count, Number(byId("generation-step").value) + offset)));
    renderGeneration();
  };
  byId("generation-step").addEventListener("input", () => { stopPlayback(); renderGeneration(); });
  byId("generation-prev").addEventListener("click", () => { stopPlayback(); move(-1); });
  byId("generation-next").addEventListener("click", () => { stopPlayback(); move(1); });
  byId("generation-reset").addEventListener("click", () => {
    stopPlayback(); byId("generation-step").value = "1"; renderGeneration();
  });
  byId("generation-play").addEventListener("click", () => {
    if (timer !== undefined) { stopPlayback(); return; }
    if (Number(byId("generation-step").value) === generation.count) byId("generation-step").value = "1";
    renderGeneration();
    byId("generation-play").textContent = "Pause steps";
    timer = setInterval(() => move(1), 1000);
  });
  renderGeneration();

  const tabs = [...document.querySelectorAll('[role="tab"]')];
  function activate(tab) {
    stopPlayback();
    tabs.forEach(item => {
      const selected = item === tab;
      item.setAttribute("aria-selected", String(selected));
      item.tabIndex = selected ? 0 : -1;
      byId(item.getAttribute("aria-controls")).hidden = !selected;
    });
  }
  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => activate(tab));
    tab.addEventListener("keydown", event => {
      const target = { ArrowRight: (index + 1) % tabs.length, ArrowLeft: (index + tabs.length - 1) % tabs.length, Home: 0, End: tabs.length - 1 }[event.key];
      if (target === undefined) return;
      event.preventDefault();
      activate(tabs[target]);
      tabs[target].focus();
    });
  });
  byId("microscope-explorer").hidden = false;
  byId("microscope-status").textContent = "";
}

main().catch(error => {
  console.error("GPT microscope could not be displayed", error);
  byId("microscope-explorer").hidden = true;
  const status = byId("microscope-status");
  status.setAttribute("role", "alert");
  status.textContent = `Unable to load the microscope: ${error.message}. The recorded trace download and other lab sections remain available.`;
});
