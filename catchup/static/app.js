const PLAYER_ONE = 0;
const PLAYER_TWO = 1;
const EMPTY = -1;

const board = document.querySelector("#board");
const finishButton = document.querySelector("#finish-button");
const undoButton = document.querySelector("#undo-button");
const resetButton = document.querySelector("#reset-button");
const suggestionButton = document.querySelector("#suggestion-button");
const simulationInput = document.querySelector("#simulation-input");
const suggestionTabs = document.querySelector("#suggestion-tabs");
const puctControls = document.querySelector(".puct-controls");
const puctPriorSelect = document.querySelector("#puct-prior-select");
const puctRolloutSelect = document.querySelector("#puct-rollout-select");
const neuralControls = document.querySelector(".neural-controls");
const neuralModelSelect = document.querySelector("#neural-model-select");
const suggestionOutput = document.querySelector("#suggestion-output");
const suggestionChoices = document.querySelector("#suggestion-choices");
const statusLine = document.querySelector("#status-line");
const blueGroups = document.querySelector("#blue-groups");
const whiteGroups = document.querySelector("#white-groups");
const emptySummary = document.querySelector("#empty-summary");
const emptyRegions = document.querySelector("#empty-regions");
const message = document.querySelector("#message");

const cellSize = 34;
const sqrt3 = Math.sqrt(3);
const minBoardZoom = 0.4;
const maxBoardZoom = 3;
const defaultNeuralModel = "data/models/directional_cnn_h128_tanh_margin_random_init_adamw_wd1e4_iter_0080_npuct200_replay_mlx.safetensors";
const suggestionProfiles = [
  {
    id: "uct",
    label: "UCT",
    solver: "mcts",
    simulations: 100,
    puctPrior: "heuristic",
    puctRollout: "biased",
    neuralModel: defaultNeuralModel,
    suggestionText: "",
    suggestionRows: [],
  },
  {
    id: "puct",
    label: "PUCT",
    solver: "puct",
    simulations: 100,
    puctPrior: "heuristic",
    puctRollout: "biased",
    neuralModel: defaultNeuralModel,
    suggestionText: "",
    suggestionRows: [],
  },
  {
    id: "neural",
    label: "Neural",
    solver: "neural-puct",
    simulations: 100,
    puctPrior: "heuristic",
    puctRollout: "biased",
    neuralModel: defaultNeuralModel,
    suggestionText: "",
    suggestionRows: [],
  },
];
let state = null;
let activeSuggestionProfileId = "uct";
let suggestionLoading = false;
let suggestionLoadingProfileId = null;
let gestureStartZoom = 1;
let neuralModels = [
  {
    label: defaultNeuralModel.split("/").at(-1),
    path: defaultNeuralModel,
  },
];
const boardView = {
  base: null,
  view: null,
  zoom: 0.75,
  centerX: null,
  centerY: null,
};

function activeSuggestionProfile() {
  return suggestionProfiles.find((profile) => profile.id === activeSuggestionProfileId) ?? suggestionProfiles[0];
}

function clearSuggestionResults() {
  for (const profile of suggestionProfiles) {
    profile.suggestionText = "";
    profile.suggestionRows = [];
  }
}

function applyActiveProfileToControls() {
  const profile = activeSuggestionProfile();
  simulationInput.value = String(profile.simulations);
  puctPriorSelect.value = profile.puctPrior;
  puctRolloutSelect.value = profile.puctRollout;
  renderNeuralModelOptions(profile.neuralModel);
}

function saveActiveControlsToProfile() {
  const profile = activeSuggestionProfile();
  profile.simulations = Math.max(1, Number.parseInt(simulationInput.value, 10) || 1);
  profile.puctPrior = puctPriorSelect.value;
  profile.puctRollout = puctRolloutSelect.value;
  profile.neuralModel = neuralModelSelect.value;
}

function renderNeuralModelOptions(selectedPath) {
  neuralModelSelect.replaceChildren();
  const hasSelected = neuralModels.some((model) => model.path === selectedPath);
  const models = hasSelected
    ? neuralModels
    : [{ label: selectedPath.split("/").at(-1), path: selectedPath }, ...neuralModels];

  for (const model of models) {
    const option = document.createElement("option");
    option.value = model.path;
    option.textContent = model.label;
    option.title = model.path;
    neuralModelSelect.append(option);
  }
  neuralModelSelect.value = selectedPath;
}

async function loadNeuralModels() {
  try {
    const response = await fetch("/api/models");
    const payload = await response.json();
    if (Array.isArray(payload.models) && payload.models.length > 0) {
      neuralModels = payload.models;
    }
  } finally {
    applyActiveProfileToControls();
  }
}

function renderSuggestionTabs() {
  suggestionTabs.replaceChildren();

  for (const profile of suggestionProfiles) {
    const tab = document.createElement("button");
    const selected = profile.id === activeSuggestionProfileId;
    tab.className = "suggestion-tab";
    tab.type = "button";
    tab.role = "tab";
    tab.ariaSelected = selected ? "true" : "false";
    tab.classList.toggle("active", selected);
    tab.disabled = suggestionLoading;
    tab.textContent = profile.label;
    tab.addEventListener("click", () => {
      saveActiveControlsToProfile();
      activeSuggestionProfileId = profile.id;
      applyActiveProfileToControls();
      render();
    });
    suggestionTabs.append(tab);
  }
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function setBoardBaseViewBox(width, height) {
  const previous = boardView.base;
  boardView.base = { x: 0, y: 0, width, height };
  if (boardView.centerX === null || boardView.centerY === null) {
    boardView.centerX = width / 2;
    boardView.centerY = height / 2;
  } else if (previous && (previous.width !== width || previous.height !== height)) {
    boardView.centerX = (boardView.centerX / previous.width) * width;
    boardView.centerY = (boardView.centerY / previous.height) * height;
  }
  applyBoardViewBox();
}

function applyBoardViewBox() {
  if (!boardView.base) return;

  const base = boardView.base;
  const viewWidth = base.width / boardView.zoom;
  const viewHeight = base.height / boardView.zoom;
  let centerX = boardView.centerX ?? base.width / 2;
  let centerY = boardView.centerY ?? base.height / 2;

  if (viewWidth <= base.width) {
    centerX = clamp(centerX, base.x + viewWidth / 2, base.x + base.width - viewWidth / 2);
  } else {
    centerX = base.x + base.width / 2;
  }
  if (viewHeight <= base.height) {
    centerY = clamp(centerY, base.y + viewHeight / 2, base.y + base.height - viewHeight / 2);
  } else {
    centerY = base.y + viewHeight / 2;
  }

  const view = {
    x: centerX - viewWidth / 2,
    y: centerY - viewHeight / 2,
    width: viewWidth,
    height: viewHeight,
  };
  boardView.centerX = centerX;
  boardView.centerY = centerY;
  boardView.view = view;
  board.setAttribute("viewBox", `${view.x} ${view.y} ${view.width} ${view.height}`);
}

function boardPointFromClient(clientX, clientY) {
  const rect = board.getBoundingClientRect();
  const view = boardView.view ?? boardView.base;
  if (!view || rect.width === 0 || rect.height === 0) return null;

  const nx = clamp((clientX - rect.left) / rect.width, 0, 1);
  const ny = clamp((clientY - rect.top) / rect.height, 0, 1);
  return {
    x: view.x + nx * view.width,
    y: view.y + ny * view.height,
    nx,
    ny,
  };
}

function setBoardZoomAt(clientX, clientY, zoom) {
  if (!boardView.base) return;

  const point = boardPointFromClient(clientX, clientY);
  if (!point) return;

  const nextZoom = clamp(zoom, minBoardZoom, maxBoardZoom);
  const nextWidth = boardView.base.width / nextZoom;
  const nextHeight = boardView.base.height / nextZoom;
  boardView.zoom = nextZoom;
  boardView.centerX = point.x + (0.5 - point.nx) * nextWidth;
  boardView.centerY = point.y + (0.5 - point.ny) * nextHeight;
  applyBoardViewBox();
}

function zoomBoardBy(clientX, clientY, factor) {
  setBoardZoomAt(clientX, clientY, boardView.zoom * factor);
}

function handleBoardWheel(event) {
  if (!event.ctrlKey && !event.metaKey) return;
  event.preventDefault();
  zoomBoardBy(event.clientX, event.clientY, Math.exp(-event.deltaY * 0.001));
}

function handleGestureStart(event) {
  event.preventDefault();
  gestureStartZoom = boardView.zoom;
}

function handleGestureChange(event) {
  event.preventDefault();
  setBoardZoomAt(event.clientX, event.clientY, gestureStartZoom * event.scale);
}

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
    clearSuggestionResults();
    render();
    message.textContent = payload.error;
    return;
  }
  state = payload;
  clearSuggestionResults();
  render();
}

async function loadState() {
  const response = await fetch("/api/state");
  state = await response.json();
  clearSuggestionResults();
  render();
}

async function requestSuggestion() {
  if (!state || state.terminal || suggestionLoading) return;

  const requestedSimulations = Math.max(1, Number.parseInt(simulationInput.value, 10) || 1);
  simulationInput.value = String(requestedSimulations);
  saveActiveControlsToProfile();
  const profile = activeSuggestionProfile();
  suggestionLoading = true;
  suggestionLoadingProfileId = profile.id;
  render();
  try {
    const response = await fetch("/api/suggest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        simulations: requestedSimulations,
        solver: profile.solver,
        puct_prior: profile.puctPrior,
        puct_rollout: profile.puctRollout,
        neural_model: profile.neuralModel,
        neural_backend: "mlx",
      }),
    });
    const payload = await response.json();
    state = payload.state;
    if (!response.ok) {
      profile.suggestionText = payload.error;
      profile.suggestionRows = [];
      return;
    }

    const suggestion = payload.suggestion;
    const engine = suggestion.engine ? `, ${suggestion.engine}` : "";
    const modelLabel = profile.solver === "neural-puct" ? `, ${profile.neuralModel.split("/").at(-1)}` : "";
    profile.suggestionText = `${suggestion.player_name}: ${suggestion.label} (${suggestion.simulations} simulations${modelLabel}${engine})`;
    profile.suggestionRows = payload.choices;
  } finally {
    suggestionLoading = false;
    suggestionLoadingProfileId = null;
    render();
  }
}

function renderBoard() {
  const centers = state.board.cells.map((cell) => ({ cell, ...centerFor(cell) }));
  const minX = Math.min(...centers.map((item) => item.x));
  const maxX = Math.max(...centers.map((item) => item.x));
  const minY = Math.min(...centers.map((item) => item.y));
  const maxY = Math.max(...centers.map((item) => item.y));
  const marginX = 48;
  const marginY = 8;
  const width = maxX - minX + marginX * 2;
  const height = maxY - minY + marginY * 2;
  const legal = legalSet();
  const selected = selectedSet();

  board.replaceChildren();

  for (const item of centers) {
    const cx = item.x - minX + marginX;
    const cy = item.y - minY + marginY;
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
  setBoardBaseViewBox(width, height);
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
  renderSuggestionTabs();

  const profile = activeSuggestionProfile();
  const isPuct = profile.solver === "puct";
  const isNeuralPuct = profile.solver === "neural-puct";
  const isCurrentProfileLoading = suggestionLoading && suggestionLoadingProfileId === profile.id;
  puctControls.hidden = !isPuct;
  neuralControls.hidden = !isNeuralPuct;
  suggestionButton.disabled = state.terminal || suggestionLoading;
  simulationInput.disabled = suggestionLoading;
  puctPriorSelect.disabled = suggestionLoading || !isPuct;
  puctRolloutSelect.disabled = suggestionLoading || !isPuct;
  neuralModelSelect.disabled = suggestionLoading || !isNeuralPuct;
  suggestionOutput.textContent = isCurrentProfileLoading
    ? `Evaluating ${profile.label}...`
    : profile.suggestionText;
  suggestionChoices.replaceChildren();

  if (isCurrentProfileLoading || profile.suggestionRows.length === 0) return;

  for (const choice of profile.suggestionRows.slice(0, 8)) {
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
    ? `Turn ${state.completed_turns}: game over, ${state.winner} wins.`
    : `Turn ${state.completed_turns}: ${state.current_player_name} to claim ${Math.max(0, state.max_claims - selectedCount)} more, or finish when available.`;
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
puctPriorSelect.addEventListener("change", () => saveActiveControlsToProfile());
puctRolloutSelect.addEventListener("change", () => saveActiveControlsToProfile());
simulationInput.addEventListener("input", () => saveActiveControlsToProfile());
neuralModelSelect.addEventListener("change", () => saveActiveControlsToProfile());
board.addEventListener("wheel", handleBoardWheel, { passive: false });
board.addEventListener("gesturestart", handleGestureStart);
board.addEventListener("gesturechange", handleGestureChange);
loadNeuralModels();
loadState();
