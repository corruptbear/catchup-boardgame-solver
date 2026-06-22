const PLAYER_ONE = 0;
const PLAYER_TWO = 1;
const EMPTY = -1;

const board = document.querySelector("#board");
const finishButton = document.querySelector("#finish-button");
const undoButton = document.querySelector("#undo-button");
const resetButton = document.querySelector("#reset-button");
const suggestionButton = document.querySelector("#suggestion-button");
const simulationInput = document.querySelector("#simulation-input");
const suggestionOutput = document.querySelector("#suggestion-output");
const suggestionChoices = document.querySelector("#suggestion-choices");
const statusLine = document.querySelector("#status-line");
const currentPlayer = document.querySelector("#current-player");
const claimCount = document.querySelector("#claim-count");
const emptyCount = document.querySelector("#empty-count");
const turnCount = document.querySelector("#turn-count");
const blueLargest = document.querySelector("#blue-largest");
const whiteLargest = document.querySelector("#white-largest");
const blueGroups = document.querySelector("#blue-groups");
const whiteGroups = document.querySelector("#white-groups");
const emptySummary = document.querySelector("#empty-summary");
const emptyRegions = document.querySelector("#empty-regions");
const message = document.querySelector("#message");

const cellSize = 34;
const sqrt3 = Math.sqrt(3);
let state = null;
let suggestionText = "";
let suggestionRows = [];
let suggestionLoading = false;

function centerFor(cell) {
  return {
    x: cellSize * sqrt3 * (cell.q + cell.r / 2),
    y: cellSize * 1.5 * cell.r,
  };
}

function hexPoints(cx, cy) {
  const points = [];
  for (let corner = 0; corner < 6; corner += 1) {
    const angle = Math.PI / 6 + corner * Math.PI / 3;
    points.push(`${(cx + cellSize * Math.cos(angle)).toFixed(2)},${(cy + cellSize * Math.sin(angle)).toFixed(2)}`);
  }
  return points.join(" ");
}

function ownerClass(owner) {
  if (owner === PLAYER_ONE) return "blue";
  if (owner === PLAYER_TWO) return "white";
  return "empty";
}

function selectedSet() {
  return new Set(state.selected);
}

function legalSet() {
  return new Set(state.legal_actions);
}

async function postJson(url, body = {}) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const payload = await response.json();
  if (!response.ok) {
    state = payload.state;
    suggestionText = "";
    suggestionRows = [];
    render();
    message.textContent = payload.error;
    return;
  }
  state = payload;
  suggestionText = "";
  suggestionRows = [];
  render();
}

async function loadState() {
  const response = await fetch("/api/state");
  state = await response.json();
  suggestionText = "";
  suggestionRows = [];
  render();
}

async function requestSuggestion() {
  if (!state || state.terminal || suggestionLoading) return;

  const requestedSimulations = Math.max(1, Number.parseInt(simulationInput.value, 10) || 1);
  simulationInput.value = String(requestedSimulations);
  suggestionLoading = true;
  render();
  try {
    const response = await fetch("/api/suggest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ simulations: requestedSimulations }),
    });
    const payload = await response.json();
    state = payload.state;
    if (!response.ok) {
      suggestionText = payload.error;
      suggestionRows = [];
      return;
    }

    const suggestion = payload.suggestion;
    const engine = suggestion.engine ? `, ${suggestion.engine}` : "";
    suggestionText = `${suggestion.player_name}: ${suggestion.label} (${suggestion.simulations} simulations${engine})`;
    suggestionRows = payload.choices;
  } finally {
    suggestionLoading = false;
    render();
  }
}

function renderBoard() {
  const centers = state.board.cells.map((cell) => ({ cell, ...centerFor(cell) }));
  const minX = Math.min(...centers.map((item) => item.x));
  const maxX = Math.max(...centers.map((item) => item.x));
  const minY = Math.min(...centers.map((item) => item.y));
  const maxY = Math.max(...centers.map((item) => item.y));
  const margin = 48;
  const width = maxX - minX + margin * 2;
  const height = maxY - minY + margin * 2;
  const legal = legalSet();
  const selected = selectedSet();

  board.setAttribute("viewBox", `0 0 ${width} ${height}`);
  board.replaceChildren();

  for (const item of centers) {
    const cx = item.x - minX + margin;
    const cy = item.y - minY + margin;
    const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
    const polygon = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
    const coordLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
    const indexLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
    const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
    const isLegal = legal.has(item.cell.index);

    group.classList.add("cell", ownerClass(item.cell.owner));
    group.classList.toggle("legal", isLegal);
    group.classList.toggle("disabled", !isLegal);
    group.classList.toggle("selected", selected.has(item.cell.index));
    group.dataset.index = String(item.cell.index);

    polygon.setAttribute("points", hexPoints(cx, cy));
    coordLabel.setAttribute("x", cx);
    coordLabel.setAttribute("y", cy + 14);
    coordLabel.setAttribute("class", "coord-label");
    coordLabel.textContent = `(${item.cell.q},${item.cell.r})`;
    indexLabel.setAttribute("x", cx);
    indexLabel.setAttribute("y", cy - 3);
    indexLabel.setAttribute("class", "index-label");
    indexLabel.textContent = `#${item.cell.index}`;
    title.textContent = `Cell ${item.cell.index}, coordinate (${item.cell.q}, ${item.cell.r})`;

    if (isLegal) {
      group.addEventListener("click", () => postJson("/api/action", { action: item.cell.index }));
    }

    group.append(title, polygon, coordLabel, indexLabel);
    board.append(group);
  }
}

function renderGroups(container, sizes) {
  container.replaceChildren();
  if (sizes.length === 0) {
    const empty = document.createElement("span");
    empty.className = "group-chip";
    empty.textContent = "0";
    container.append(empty);
    return;
  }
  for (const size of sizes) {
    const chip = document.createElement("span");
    chip.className = "group-chip";
    chip.textContent = String(size);
    container.append(chip);
  }
}

function formatCells(cells) {
  return cells.map((cell) => `#${cell}`).join(" ");
}

function formatBoundaries(components) {
  if (components.length === 0) return "none";
  return components.map((component) => `#${component.root}(${component.size})`).join(" ");
}

function renderEmptyRegions() {
  emptySummary.textContent = `${state.empty_components.length} region${state.empty_components.length === 1 ? "" : "s"}`;
  emptyRegions.replaceChildren();

  for (const region of state.empty_components) {
    const wrapper = document.createElement("div");
    const title = document.createElement("div");
    const root = document.createElement("span");
    const size = document.createElement("span");
    const cells = document.createElement("div");
    const blue = document.createElement("div");
    const white = document.createElement("div");

    wrapper.className = "empty-region";
    title.className = "region-title";
    root.textContent = `Root #${region.root}`;
    size.textContent = `size ${region.size}`;
    cells.className = "region-cells";
    cells.textContent = formatCells(region.cells);
    blue.className = "boundary-row";
    blue.innerHTML = `<strong>Blue:</strong> ${formatBoundaries(region.blue)}`;
    white.className = "boundary-row";
    white.innerHTML = `<strong>White:</strong> ${formatBoundaries(region.white)}`;

    title.append(root, size);
    wrapper.append(title, cells, blue, white);
    emptyRegions.append(wrapper);
  }
}

function renderSuggestion() {
  suggestionButton.disabled = state.terminal || suggestionLoading;
  simulationInput.disabled = suggestionLoading;
  suggestionOutput.textContent = suggestionLoading ? "Evaluating..." : suggestionText;
  suggestionChoices.replaceChildren();

  if (suggestionLoading || suggestionRows.length === 0) return;

  for (const choice of suggestionRows.slice(0, 8)) {
    const row = document.createElement("div");
    const label = document.createElement("span");
    const visits = document.createElement("span");

    row.className = "suggestion-choice";
    label.textContent = choice.label;
    visits.textContent = `${choice.visits} visit${choice.visits === 1 ? "" : "s"}`;
    row.append(label, visits);
    suggestionChoices.append(row);
  }
}

function renderStats() {
  const blue = state.players.find((player) => player.id === PLAYER_ONE);
  const white = state.players.find((player) => player.id === PLAYER_TWO);
  const selectedCount = state.selected.length;

  statusLine.textContent = state.terminal
    ? `Game over: ${state.winner}`
    : `${state.current_player_name} to claim ${Math.max(0, state.max_claims - selectedCount)} more, or finish when available.`;
  currentPlayer.textContent = state.current_player_name;
  claimCount.textContent = `${selectedCount}/${state.max_claims}`;
  emptyCount.textContent = state.empty_count;
  turnCount.textContent = state.completed_turns;
  blueLargest.textContent = `Largest: ${blue.largest_group}`;
  whiteLargest.textContent = `Largest: ${white.largest_group}`;
  message.textContent = state.message;

  renderGroups(blueGroups, blue.group_sizes);
  renderGroups(whiteGroups, white.group_sizes);
  renderEmptyRegions();
  renderSuggestion();

  finishButton.disabled = !state.legal_actions.includes(state.finish_action);
}

function render() {
  if (!state) return;
  renderBoard();
  renderStats();
}

finishButton.addEventListener("click", () => postJson("/api/action", { action: state.finish_action }));
undoButton.addEventListener("click", () => postJson("/api/undo"));
resetButton.addEventListener("click", () => postJson("/api/reset"));
suggestionButton.addEventListener("click", () => requestSuggestion());
loadState();
