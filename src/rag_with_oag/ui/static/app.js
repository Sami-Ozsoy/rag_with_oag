const $ = (id) => document.getElementById(id);
const tokenKey = "ontology-studio-token";
const reportHeightKey = "ontology-studio-report-height";
const openLinkMessage = "Open the link printed in the terminal.";
const token = readToken();
const jobButtons = [...document.querySelectorAll("[data-job]")];
const tabs = [...document.querySelectorAll('[role="tab"]')];
const jobLabels = new Map(jobButtons.map((button) => [button, button.textContent]));
const propertyTypes = ["string", "integer", "number", "boolean", "date", "enum"];
const statusLabels = {
  covered: "Covered",
  text: "Answered from text",
  not_covered: "Not covered",
  invalid_path: "Invalid path",
  too_many_hops: "Too many hops",
};

let workspace;
let ontology = null;
let selection = null;
let dirty = false;
let revision = 0;
let yamlText = "";
let yamlDraft = null;
let yamlOpen = false;
let validationTimer;
let validation;
let errors = [];
let previewReport = null;
let previewOutdated = false;
let questionReport = null;
let activeQuestion = null;
let pathTimer;
let activeJob = null;
let saving = false;
let toastTimer;
let approvalQueue = Promise.resolve();

function storedValue(storage, key) {
  try { return window[storage].getItem(key); } catch { return null; }
}

function storeValue(storage, key, value) {
  try { window[storage].setItem(key, value); } catch { /* Storage can be disabled. */ }
}

function readToken() {
  const fromLink = new URLSearchParams(location.hash.slice(1)).get("token");
  if (fromLink === null) return storedValue("sessionStorage", tokenKey) ?? "";
  storeValue("sessionStorage", tokenKey, fromLink);
  history.replaceState(null, "", location.pathname);
  return fromLink;
}

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text != null) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function button(label, action, className) {
  const node = element("button", label, className);
  node.type = "button";
  node.addEventListener("click", action);
  return node;
}

function appendList(container, entries) {
  const list = element("ul");
  for (const entry of entries) list.append(element("li", entry));
  container.append(list);
}

class ApiError extends Error {
  constructor(status, payload) {
    super(payload.errors?.join("\n") || payload.detail || `Request failed (${status})`);
    this.status = status;
    this.payload = payload;
  }
}

async function api(path, body) {
  const headers = { "X-Studio-Token": token };
  const options = { headers };
  if (body !== undefined) {
    headers["Content-Type"] = "application/json";
    options.method = "POST";
    options.body = JSON.stringify(body);
  }
  const response = await fetch(`/api/${path}`, options);
  if (response.status === 401) throw new Error(openLinkMessage);
  const payload = await response.json();
  if (!response.ok) throw new ApiError(response.status, payload);
  return payload;
}

function showError(error) {
  $("banner-text").textContent = error instanceof Error ? error.message : error;
  $("banner").hidden = false;
}

function toast(message) {
  clearTimeout(toastTimer);
  $("toast").textContent = message;
  $("toast").hidden = false;
  toastTimer = setTimeout(() => { $("toast").hidden = true; }, 4200);
}

function askApproval(title, message, action) {
  const choice = approvalQueue.then(() => new Promise((resolve) => {
    const dialog = $("approval-dialog");
    $("approval-title").textContent = title;
    $("approval-message").textContent = message;
    $("approval-action").textContent = action;
    dialog.returnValue = "";
    dialog.showModal();
    dialog.addEventListener("close", () => resolve(dialog.returnValue === "proceed"), { once: true });
  }));
  approvalQueue = choice.catch(() => {});
  return choice;
}

function setDirty(value) {
  dirty = value;
  $("dirty").hidden = !dirty;
}

function showTab(selected, focus = false) {
  for (const tab of tabs) {
    const active = tab.id === `tab-${selected}`;
    tab.setAttribute("aria-selected", active);
    tab.tabIndex = active ? 0 : -1;
    $(tab.getAttribute("aria-controls")).hidden = !active;
    if (active && focus) tab.focus();
  }
}

function setupReportResize() {
  const studio = $("studio");
  const handle = $("report-resize");
  const panel = $("workbench");
  const panelHeight = () => panel.offsetHeight;
  let drag = null;

  function bounds() {
    const styles = getComputedStyle(studio);
    const min = parseFloat(styles.getPropertyValue("--reports-min"));
    const workspaceMin = studio.clientHeight * Number(styles.getPropertyValue("--workspace-share"));
    const padding = parseFloat(styles.paddingTop) + parseFloat(styles.paddingBottom);
    const max = Math.floor(studio.clientHeight - padding - handle.offsetHeight - workspaceMin);
    return { min, max: Math.max(min, max) };
  }

  function updateHandle() {
    const { min, max } = bounds();
    handle.setAttribute("aria-valuemin", min);
    handle.setAttribute("aria-valuemax", max);
    handle.setAttribute("aria-valuenow", panelHeight());
    handle.setAttribute("aria-valuetext", `${panelHeight()} pixels`);
  }

  function setHeight(height, persist = false) {
    const { min, max } = bounds();
    const next = Math.round(Math.min(max, Math.max(min, height)));
    studio.style.setProperty("--reports-height", `${next}px`);
    if (persist) storeValue("localStorage", reportHeightKey, String(next));
    updateHandle();
  }

  const saved = Number(storedValue("localStorage", reportHeightKey));
  if (Number.isFinite(saved) && saved > 0) setHeight(saved);
  updateHandle();

  handle.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 || drag) return;
    handle.setPointerCapture(event.pointerId);
    drag = { id: event.pointerId, startY: event.clientY, startHeight: panelHeight() };
  });
  handle.addEventListener("pointermove", (event) => {
    if (drag?.id === event.pointerId) setHeight(drag.startHeight + drag.startY - event.clientY);
  });
  handle.addEventListener("lostpointercapture", (event) => {
    if (drag?.id !== event.pointerId) return;
    drag = null;
    storeValue("localStorage", reportHeightKey, String(panelHeight()));
  });
  handle.addEventListener("keydown", (event) => {
    const { min, max } = bounds();
    const heights = { ArrowUp: panelHeight() + 20, ArrowDown: panelHeight() - 20, Home: min, End: max };
    if (heights[event.key] === undefined) return;
    event.preventDefault();
    setHeight(heights[event.key], true);
  });

  const resizeObserver = new ResizeObserver(updateHandle);
  resizeObserver.observe(studio);
  resizeObserver.observe(panel);
}

function plural(number, noun) {
  return `${number} ${noun}${number === 1 ? "" : "s"}`;
}

function updateControls() {
  const ready = !!workspace;
  const hasSchema = !!ontology || yamlDraft !== null;
  $("layout").disabled = !ontology?.entities.length;
  $("add-entity").disabled = !ready;
  $("add-relation").disabled = !ontology?.entities.length;
  $("show-yaml").disabled = !hasSchema;
  $("save").disabled = !hasSchema || saving;
  for (const button of jobButtons) {
    const kind = button.dataset.job;
    const running = activeJob?.kind === kind;
    const needsSchema = kind !== "suggest" && !hasSchema;
    const needsQuestions = kind === "check-questions" && !questions().length;
    button.disabled = !workspace?.model || !!activeJob || needsSchema || needsQuestions;
    button.title = workspace?.model_error || (needsSchema ? "Create or load an ontology first." :
      needsQuestions ? "Enter at least one question." : "");
    if (running) {
      const elapsed = Math.floor((performance.now() - activeJob.started) / 1000);
      button.replaceChildren(element("span", "", "spinner"), `${jobLabels.get(button)} · ${elapsed}s`);
      button.setAttribute("aria-busy", "true");
    } else {
      button.textContent = jobLabels.get(button);
      button.removeAttribute("aria-busy");
    }
  }
}

function updateProblems(next) {
  errors = next;
  for (const list of [$("problems"), $("yaml-errors")]) {
    list.replaceChildren(...errors.map((error) => element("li", error)));
  }
  $("yaml-errors").hidden = !errors.length;
  $("problems-empty").hidden = !!errors.length;
  $("problem-count").textContent = errors.length;
  $("problem-count").hidden = !errors.length;
  $("yaml-editor").setAttribute("aria-invalid", !!errors.length);
}

function scheduleValidation() {
  revision += 1;
  setDirty(true);
  clearTimeout(validationTimer);
  validationTimer = setTimeout(() => {
    validate().catch(showError);
  }, 400);
  updateControls();
}

async function validate() {
  clearTimeout(validationTimer);
  const current = revision;
  if (validation?.revision !== current) {
    const fromYaml = yamlDraft !== null;
    const request = fromYaml
      ? api("ontology/parse", { yaml: yamlDraft })
      : api("ontology/render", { ontology });
    const promise = request.then((result) => {
      if (revision !== current) return false;
      updateProblems(result.errors ?? []);
      if (!result.ok) return false;
      if (fromYaml) {
        yamlText = yamlDraft;
        replaceOntology(result.ontology);
      } else {
        yamlText = result.yaml;
        if (yamlOpen) $("yaml-editor").value = yamlText;
      }
      return true;
    }).catch((error) => {
      if (validation?.revision === current) validation = undefined;
      if (revision === current) throw error;
      return false;
    });
    validation = { revision: current, promise };
  }
  const valid = await validation.promise;
  return valid && revision === current;
}

async function currentYaml() {
  if (await validate()) return yamlText;
  if (errors.length) showTab("problems");
  return null;
}

function structureOf(schema) {
  const entities = (schema?.entities ?? []).map((entity) => entity.name).sort();
  const relations = (schema?.relations ?? []).map(({ name, source, target }) =>
    JSON.stringify([name, source, target])).sort();
  return JSON.stringify([entities, relations]);
}

function invalidatePreview() {
  if (previewReport) {
    previewOutdated = true;
    $("workbench-status").textContent = "Preview is out of date";
    renderPreview();
    renderInspectorPreview();
  }
  $("edge-tooltip").hidden = true;
  clearPath();
}

function replaceOntology(next) {
  const oldStructure = structureOf(ontology);
  const selectedName = selection?.item.name;
  ontology = next;
  const selectedItems = selection?.kind === "entity" ? next.entities : next.relations;
  const item = selectedItems.find((candidate) => candidate.name === selectedName);
  selection = item ? { kind: selection.kind, item } : null;
  if (structureOf(ontology) !== oldStructure) invalidatePreview();
  updateSchema();
}

function changeSchema(edit, redrawInspector = false) {
  const oldStructure = structureOf(ontology);
  edit();
  yamlDraft = null;
  scheduleValidation();
  if (structureOf(ontology) !== oldStructure) invalidatePreview();
  updateSchema(redrawInspector);
}

function showYaml(open) {
  yamlOpen = open;
  $("yaml-panel").hidden = !open;
  $("schema-panel").hidden = open;
  $("inspector-panel").hidden = open;
  $("show-yaml").setAttribute("aria-pressed", open);
  $("show-yaml").textContent = open ? "Schema" : "YAML";
  if (open) {
    $("yaml-editor").value = yamlDraft ?? yamlText;
    $("yaml-editor").focus();
    if (yamlDraft === null) validate().catch(showError);
  } else {
    cy.resize();
    drawGraph();
    fitGraph();
  }
}

function setPath(path) {
  workspace.ontology_path = path;
  $("yaml-file").textContent = path ?? "ontology.yaml";
}

function graphStyles() {
  const colors = getComputedStyle(document.documentElement);
  const color = (variable) => colors.getPropertyValue(variable).trim();
  return [
    { selector: "node", style: {
      shape: "round-rectangle", width: "data(width)", height: 51,
      "background-color": color("--node"), "border-width": 1.6, "border-color": color("--node-line"),
      label: "data(label)", color: color("--text"), "font-family": "system-ui, sans-serif",
      "font-weight": 600, "font-size": 12, "text-valign": "center", "text-halign": "center",
      "text-wrap": "wrap", "text-max-width": 185,
    } },
    { selector: "edge", style: {
      "curve-style": "bezier",
      width: 1.8, "line-color": color("--edge"), "target-arrow-color": color("--edge"),
      "target-arrow-shape": "triangle", label: "data(label)", color: color("--text"),
      "font-family": "system-ui, sans-serif", "font-size": 11, "font-weight": 600,
      "text-background-color": color("--canvas"), "text-background-opacity": 1,
      "text-background-padding": 3, "text-background-shape": "roundrectangle",
    } },
    { selector: "edge:loop", style: {
      "control-point-step-size": 110, "loop-direction": "0deg", "loop-sweep": "-90deg",
    } },
    { selector: "node.selected-type, node.path-member", style: {
      "border-color": color("--accent"), "border-width": 3, "background-color": color("--accent-bg"),
    } },
    { selector: "edge.selected-type, edge.path-member", style: {
      width: 3, "line-color": color("--accent"), "target-arrow-color": color("--accent"),
    } },
    { selector: ".unused", style: { opacity: .48 } },
    { selector: ".path-dim", style: { opacity: .2 } },
    { selector: ".path-member", style: { opacity: 1 } },
    { selector: "node.low-key-fill", style: { "border-color": color("--warning"), "border-width": 3 } },
    { selector: "edge.mismatch", style: {
      width: 3, "line-color": color("--error"), "target-arrow-color": color("--error"), opacity: 1,
    } },
    { selector: "edge.mismatch.path-dim", style: { opacity: .2 } },
  ];
}

const cy = window.cytoscape({
  container: $("graph"),
  elements: [],
  style: graphStyles(),
  layout: { name: "preset" },
  minZoom: 0.25,
  maxZoom: 2.5,
  boxSelectionEnabled: false,
  autounselectify: true,
});
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => cy.style(graphStyles()));
const graphIds = new WeakMap();
let nextId = 0;
let drawnSchema = "";
let graphItems = new Map();
let viewportMoved = false;

cy.on("dragpan scrollzoom pinchzoom", () => { viewportMoved = true; });
new ResizeObserver(([entry]) => {
  const { width, height } = entry.contentRect;
  if (yamlOpen || !width || !height) return;
  if (viewportMoved) cy.resize();
  else fitGraph();
}).observe($("graph"));

function graphId(item) {
  if (!graphIds.has(item)) graphIds.set(item, `type-${nextId++}`);
  return graphIds.get(item);
}

function rerunLayout() {
  if (!cy.nodes().length) return;
  cy.layout({ name: "cose", animate: false, fit: false, nodeRepulsion: 9500, stop: fitGraph }).run();
}

function fitGraph() {
  if (yamlOpen) return;
  viewportMoved = false;
  cy.resize();
  if (!cy.nodes().length) return;
  const padding = Math.min(44, cy.width() * .2, cy.height() * .2);
  cy.fit(cy.elements(), padding);
  if (cy.zoom() > 1.25) zoomGraph(1.25);
}

function zoomGraph(level, manual = false) {
  if (manual && level !== cy.zoom()) viewportMoved = true;
  cy.zoom({ level, renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 } });
}

function drawGraph() {
  if (yamlOpen) return;
  const entities = ontology?.entities ?? [];
  const relations = ontology?.relations ?? [];
  const fingerprint = JSON.stringify([
    entities.map((item) => [graphId(item), item.name, item.key]),
    relations.map((item) => [graphId(item), item.name, item.source, item.target]),
  ]);
  if (fingerprint !== drawnSchema) {
    drawnSchema = fingerprint;
    const oldPositions = new Map(cy.nodes().map((node) => [node.data("name"), node.position()]));
    const previousIds = new Map(cy.nodes().map((node) => [node.id(), node.position()]));
    const pan = cy.pan();
    const viewCenter = {
      x: (cy.width() / 2 - pan.x) / cy.zoom(),
      y: (cy.height() / 2 - pan.y) / cy.zoom(),
    };
    const nodeIds = new Map();
    graphItems = new Map();
    const definitions = entities.map((item, index) => {
      const id = graphId(item);
      if (!nodeIds.has(item.name)) nodeIds.set(item.name, id);
      graphItems.set(id, { kind: "entity", item });
      const label = item.key ? `${item.name || "(unnamed)"}\n${item.key}` : item.name || "(unnamed)";
      return {
        group: "nodes",
        data: { id, name: item.name, label, width: Math.min(210, Math.max(108, label.length * 7)) },
        position: previousIds.get(id) ?? oldPositions.get(item.name) ?? {
          x: viewCenter.x + (index % 5 - 2) * 34,
          y: viewCenter.y + (index % 3 - 1) * 48,
        },
      };
    });
    for (const item of relations) {
      const source = nodeIds.get(item.source);
      const target = nodeIds.get(item.target);
      if (!source || !target) continue;
      const id = graphId(item);
      graphItems.set(id, { kind: "relation", item });
      definitions.push({
        group: "edges", data: { id, name: item.name, label: item.name || "(unnamed)", source, target },
      });
    }
    cy.batch(() => {
      cy.elements().remove();
      cy.add(definitions);
    });
    if (entities.length && !oldPositions.size) rerunLayout();
  }
  paintGraph();
}

function mismatches() {
  return previewReport && !previewOutdated
    ? previewReport.dropped.filter((group) => group.kind === "endpoint_mismatch")
    : [];
}

function graphType(id, name) {
  const elements = id === "nodes" ? cy.nodes() : cy.edges();
  return elements.filter((item) => item.data("name") === name).first();
}

function paintGraph() {
  cy.batch(() => {
    cy.elements().removeClass("selected-type unused low-key-fill mismatch");
    if (selection) cy.getElementById(graphId(selection.item)).addClass("selected-type");
    if (previewReport && !previewOutdated) {
      for (const stats of previewReport.entity_types) {
        const node = graphType("nodes", stats.name);
        node.toggleClass("unused", stats.mentions === 0);
        node.toggleClass("low-key-fill", stats.key_fill_rate != null && stats.key_fill_rate < .5);
      }
      for (const stats of previewReport.relation_types) {
        graphType("edges", stats.name).toggleClass("unused", stats.count === 0);
      }
      for (const group of mismatches()) graphType("edges", group.type_name).addClass("mismatch");
    }
  });
  if (activeQuestion) paintPath(false);
  $("preview-legend").hidden = !previewReport || previewOutdated;
}

function observedPairs(group) {
  return Object.entries(group.observed).map(([pair, count]) => `${pair} (${count})`).join(", ");
}

function showEdgeTooltip(event) {
  const group = mismatches().find((item) => item.type_name === event.target.data("name"));
  if (!group || !event.renderedPosition) return;
  const tip = $("edge-tooltip");
  tip.textContent = `Endpoint mismatch · ${observedPairs(group)}`;
  tip.hidden = false;
  const { x, y } = event.renderedPosition;
  tip.style.left = `${Math.max(8, Math.min(x + 12, cy.width() - tip.offsetWidth - 8))}px`;
  tip.style.top = `${Math.max(8, Math.min(y + 12, cy.height() - tip.offsetHeight - 8))}px`;
}

cy.on("tap", "node, edge", (event) => {
  selectType(graphItems.get(event.target.id()) ?? null);
  if (event.target.isEdge()) showEdgeTooltip(event);
});
cy.on("tap", (event) => { if (event.target === cy) selectType(null); });
cy.on("mouseover", "edge", showEdgeTooltip);
cy.on("mouseout pan zoom", () => { $("edge-tooltip").hidden = true; });

function updateSchema(redrawInspector = true) {
  const entities = ontology?.entities ?? [];
  const relations = ontology?.relations ?? [];
  $("schema-count").textContent = ontology
    ? `${plural(entities.length, "entity type")} · ${plural(relations.length, "relation")}`
    : "";
  $("empty").hidden = !!entities.length;
  $("empty-title").textContent = workspace ? "No entity types yet" : "Loading workspace…";
  $("empty-actions").hidden = !workspace || !!ontology;
  $("graph-tools").hidden = !entities.length;
  $("graph").tabIndex = entities.length ? 0 : -1;
  updateControls();
  updatePicker();
  if (redrawInspector) renderInspector();
  drawGraph();
}

function selectType(next) {
  selection = next;
  updatePicker();
  renderInspector();
  paintGraph();
}

function updatePicker() {
  const picker = $("type-picker");
  picker.replaceChildren(new Option("Select a type", ""));
  for (const [kind, types] of [["entity", ontology?.entities], ["relation", ontology?.relations]]) {
    if (!types?.length) continue;
    const group = element("optgroup");
    group.label = kind === "entity" ? "Entity types" : "Relations";
    for (const item of types) group.append(new Option(item.name || "(unnamed)", graphId(item)));
    picker.append(group);
  }
  picker.value = selection ? graphId(selection.item) : "";
}

function input(value, label, update, tag = "input") {
  const field = element(tag);
  if (tag === "textarea") field.rows = 2;
  field.value = value;
  field.setAttribute("aria-label", label);
  field.addEventListener("input", () => update(field.value));
  return field;
}

function labeled(label, field) {
  const wrapper = element("label", label);
  wrapper.append(field);
  return wrapper;
}

function setOptions(field, items, value) {
  field.replaceChildren(...items.map(([key, label]) => new Option(label, key)));
  field.value = value ?? "";
}

function selectField(label, items, value, update) {
  const field = element("select");
  field.setAttribute("aria-label", label);
  setOptions(field, items, value);
  field.addEventListener("change", () => update(field.value));
  return field;
}

function freshName(prefix, types) {
  const names = new Set(types.map((item) => item.name));
  let name = prefix;
  let number = 2;
  while (names.has(name)) name = `${prefix}${number++}`;
  return name;
}

function addEntity() {
  if (!workspace) return;
  changeSchema(() => {
    ontology ??= { name: "ontology", version: "1", entities: [], relations: [], name_suffixes: [] };
    const item = { name: freshName("NewType", ontology.entities), description: "", key: null, properties: [] };
    ontology.entities.push(item);
    selection = { kind: "entity", item };
  }, true);
  $("type-name").focus();
  $("type-name").select();
}

function addRelation() {
  const endpoint = ontology.entities[0].name;
  changeSchema(() => {
    const item = {
      name: freshName("NEW_RELATION", ontology.relations), source: endpoint, target: endpoint,
      description: "", properties: [],
    };
    ontology.relations.push(item);
    selection = { kind: "relation", item };
  }, true);
  $("type-name").focus();
  $("type-name").select();
}

async function deleteType(kind, item) {
  const connected = kind === "entity" ? ontology.relations.filter((relation) =>
    relation.source === item.name || relation.target === item.name) : [];
  if (connected.length && !await askApproval(
    "Delete entity type?", `${item.name} and ${plural(connected.length, "connected relation")} will be deleted.`, "Delete",
  )) return;
  changeSchema(() => {
    if (kind === "entity") {
      ontology.entities = ontology.entities.filter((entry) => entry !== item);
      ontology.relations = ontology.relations.filter((entry) => !connected.includes(entry));
    } else {
      ontology.relations = ontology.relations.filter((entry) => entry !== item);
    }
    selection = null;
  }, true);
  $("type-picker").focus();
}

function renameType(kind, item, field) {
  const attached = () => ontology.relations.flatMap((relation) =>
    ["source", "target"].filter((end) => relation[end] === item.name).map((end) => [relation, end]));
  let endpoints = [];
  if (kind === "entity") field.addEventListener("focus", () => { endpoints = attached(); });
  field.addEventListener("input", () => {
    changeSchema(() => {
      for (const [relation, end] of endpoints) relation[end] = field.value;
      item.name = field.value;
    });
  });
}

function keyOptions(item) {
  const names = item.properties.map((prop) => [prop.name, prop.name || "(unnamed)"]);
  if (item.key && !names.some(([name]) => name === item.key)) names.push([item.key, item.key]);
  return [["", "None"], ...names];
}

function propertyRows(item, prop, keyField) {
  const row = element("tr");
  const details = element("tr", null, "property-details");
  const detail = element("td");
  detail.colSpan = 4;
  const name = input(prop.name, "Property name", (value) => {
    changeSchema(() => {
      if (keyField && item.key === prop.name) item.key = value || null;
      prop.name = value;
      if (keyField) setOptions(keyField, keyOptions(item), item.key);
    });
  });
  const enumValues = labeled("Enum values (comma-separated)", input((prop.values ?? []).join(", "), "Enum values", (value) => {
    changeSchema(() => { prop.values = value.split(",").map((part) => part.trim()).filter(Boolean); });
  }));
  enumValues.hidden = prop.type !== "enum";
  const type = selectField("Property type", propertyTypes.map((value) => [value, value]), prop.type, (value) => {
    changeSchema(() => {
      prop.type = value;
      prop.values = value === "enum" ? [] : null;
      enumValues.hidden = value !== "enum";
      enumValues.querySelector("input").value = "";
    });
  });
  const required = element("input");
  required.type = "checkbox";
  required.checked = prop.required;
  required.setAttribute("aria-label", "Required property");
  required.addEventListener("change", () => {
    changeSchema(() => { prop.required = required.checked; });
  });
  const remove = button("×", () => {
    changeSchema(() => {
      item.properties = item.properties.filter((entry) => entry !== prop);
      if (keyField && item.key === prop.name) item.key = null;
    }, true);
    $("add-property").focus();
  }, "danger");
  remove.setAttribute("aria-label", `Remove property ${prop.name}`);
  for (const field of [name, type, required, remove]) {
    const cell = element("td");
    cell.append(field);
    row.append(cell);
  }
  detail.append(enumValues, labeled("Description", input(prop.description, "Property description", (value) => {
    changeSchema(() => { prop.description = value; });
  }, "textarea")));
  details.append(detail);
  return [row, details];
}

function renderInspector() {
  const container = $("inspector-fields");
  container.replaceChildren();
  if (!selection) {
    container.append(element("p", "Select an entity type or relation to edit it.", "muted"));
    renderInspectorPreview();
    return;
  }

  const { kind, item } = selection;
  const fields = element("div", null, "inspector-form");
  const name = element("input");
  name.id = "type-name";
  name.value = item.name;
  renameType(kind, item, name);
  fields.append(labeled(kind === "entity" ? "Entity name" : "Relation name", name));

  let keyField = null;
  if (kind === "relation") {
    const endpoints = element("div", null, "endpoints");
    for (const [end, label] of [["source", "Source"], ["target", "Target"]]) {
      const options = ontology.entities.map((entry) => [entry.name, entry.name || "(unnamed)"]);
      if (!options.some(([key]) => key === item[end])) options.push([item[end], item[end]]);
      const field = selectField(label, options, item[end], (value) => {
        changeSchema(() => { item[end] = value; });
      });
      endpoints.append(labeled(label, field));
    }
    fields.append(endpoints);
  } else {
    keyField = selectField("Identifying property", keyOptions(item), item.key, (value) => {
      changeSchema(() => { item.key = value || null; });
    });
    fields.append(labeled("Identifying property", keyField));
  }

  fields.append(labeled("Description", input(item.description, "Type description", (value) => {
    changeSchema(() => { item.description = value; });
  }, "textarea")));

  const heading = element("header");
  const addProperty = button("Add property", () => {
    changeSchema(() => {
      item.properties.push({
        name: freshName("property", item.properties), type: "string", description: "",
        required: false, values: null,
      });
    }, true);
    container.querySelector('tr:nth-last-child(2) input').focus();
  });
  addProperty.id = "add-property";
  heading.append(element("h3", "Properties"), addProperty);
  fields.append(heading);

  if (item.properties.length) {
    const table = createTable(["Name", "Type", "Required", ""], []);
    table.className = "property-table";
    for (const property of item.properties) table.tBodies[0].append(...propertyRows(item, property, keyField));
    fields.append(table);
  }

  const actions = element("div", null, "inspector-actions");
  actions.append(button("Delete", () => deleteType(kind, item).catch(showError), "danger"));
  fields.append(actions);
  container.append(fields);
  renderInspectorPreview();
}

function percent(rate) {
  return rate == null ? "—" : `${Math.round(rate * 100)}%`;
}

function renderInspectorPreview() {
  const container = $("inspector-preview");
  container.replaceChildren();
  if (!selection || !previewReport || previewOutdated) return;
  const isEntity = selection.kind === "entity";
  const list = isEntity ? previewReport.entity_types : previewReport.relation_types;
  const stats = list.find((entry) => entry.name === selection.item.name);
  if (!stats) return;

  const panel = element("div", null, "inspector-stats");
  const count = isEntity
    ? `${plural(stats.mentions, "mention")} · ${stats.distinct} distinct · Key fill ${percent(stats.key_fill_rate)}`
    : plural(stats.count, "extraction");
  panel.append(element("h3", "Preview stats"), element("p", count));
  const rates = Object.entries(stats.property_fill_rates).map(([name, rate]) => `${name} ${percent(rate)}`);
  if (rates.length) panel.append(element("p", `Property fill · ${rates.join(", ")}`));
  if (stats.examples.length) panel.append(element("p", `Examples · ${stats.examples.join(", ")}`));
  const mismatch = isEntity ? null : mismatches().find((group) => group.type_name === stats.name);
  if (mismatch) panel.append(element("p", `Endpoint mismatch · ${observedPairs(mismatch)}`, "errors"));
  container.append(panel);
}

function createTable(headers, rows) {
  const table = element("table");
  const header = table.createTHead().insertRow();
  for (const name of headers) {
    const cell = element("th", name);
    cell.scope = "col";
    header.append(cell);
  }
  const body = table.createTBody();
  for (const values of rows) {
    const row = body.insertRow();
    for (const value of values) row.insertCell().textContent = value;
  }
  return table;
}

function reportTable(container, title, headers, rows) {
  if (!rows.length) return;
  const block = element("div", null, "report-block");
  const scroll = element("div", null, "table-wrap");
  scroll.append(createTable(headers, rows));
  block.append(element("h3", title), scroll);
  container.append(block);
}

function tokenSummary(usage) {
  return `Tokens · ${usage.prompt_tokens ?? 0} prompt · ${usage.completion_tokens ?? 0} completion`;
}

function renderPreview() {
  const container = $("preview-panel");
  container.replaceChildren();
  if (!previewReport) {
    container.append(element("p", "Run Preview to sample extraction from your documents.", "muted"));
    return;
  }

  if (previewOutdated) container.append(element("p", "Preview is out of date", "warning-text"));
  const report = previewReport;
  const samples = `${report.documents_sampled} of ${report.documents_available} documents · ` +
    `${plural(report.chunks_sampled, "chunk")} · ${plural(report.batches, "batch")}`;
  const coverage = `${report.chunks_with_entities}/${report.chunks_sampled} chunks with entities · ` +
    `${report.chunks_with_relations}/${report.chunks_sampled} with relations`;
  container.append(element("h3", samples), element("p", coverage, "muted"));
  reportTable(container, "Entity types", ["Type", "Mentions", "Distinct", "Key fill", "Examples"],
    report.entity_types.map((type) => [type.name, type.mentions, type.distinct,
      percent(type.key_fill_rate), type.examples.join(", ")]));
  reportTable(container, "Relations", ["Type", "Source → Target", "Extracted", "Examples"],
    report.relation_types.map((type) => [type.name, `${type.source} → ${type.target}`, type.count,
      type.examples.join(", ")]));
  reportTable(container, "Dropped items", ["Reason", "Type", "Count", "Observed pairs", "Examples"],
    report.dropped.map((group) => [group.kind.replaceAll("_", " "), group.type_name ?? "—",
      group.count, observedPairs(group), group.examples.join(", ")]));
  if (report.hints.length) {
    container.append(element("h3", "Hints"));
    appendList(container, report.hints);
  }
  if (report.failed_batches.length) {
    container.append(element("h3", "Failed batches", "errors"));
    appendList(container, report.failed_batches);
  }
  container.append(element("p", tokenSummary(report.token_usage), "muted"));
}

function questions() {
  return $("questions").value.split("\n").map((line) => line.trim()).filter(Boolean);
}

function renderQuestions() {
  const container = $("question-results");
  container.replaceChildren();
  if (!questionReport) return;
  const report = questionReport;
  const covered = report.questions.filter((entry) => entry.status === "covered").length;
  const text = report.questions.filter((entry) => entry.status === "text").length;
  const gaps = report.questions.length - covered - text;
  const summary = `Checked ${plural(report.questions.length, "question")} against ${report.ontology} ` +
    `(hop limit ${report.hops_limit}) · ${covered} covered · ${text} answered from text · ${plural(gaps, "gap")}`;
  container.append(element("p", summary));
  const results = element("div", null, "question-results");
  for (const entry of report.questions) {
    const highlightable = !!entry.start_type;
    const row = highlightable
      ? button("", () => togglePath(entry), "question-item")
      : element("div", null, "question-item");
    if (highlightable) row.setAttribute("aria-pressed", "false");
    const heading = element("span");
    heading.append(element("span", statusLabels[entry.status], `chip ${entry.status}`), entry.question);
    row.append(heading);
    if (entry.path) row.append(element("span", entry.path, "path result-detail"));
    if (entry.reason) row.append(element("span", entry.reason, "muted result-detail"));
    results.append(row);
  }
  container.append(results, element("p", tokenSummary(report.token_usage), "muted"));
}

function pathElements(entry) {
  let node = graphType("nodes", entry.start_type);
  if (!node.length) return [];
  const path = [node];
  for (const [relation, direction] of entry.steps) {
    const edge = graphType("edges", relation);
    if (!edge.length) break;
    const forward = direction === "forward";
    const origin = forward ? edge.source() : edge.target();
    if (origin.id() !== node.id()) break;
    node = forward ? edge.target() : edge.source();
    path.push(edge, node);
  }
  return path;
}

function paintPath(animate) {
  clearTimeout(pathTimer);
  const path = pathElements(activeQuestion);
  cy.elements().removeClass("path-member path-dim");
  if (!path.length) return;
  cy.elements().addClass("path-dim");
  let index = 0;
  const showNext = () => {
    path[index++].removeClass("path-dim").addClass("path-member");
    if (index < path.length) pathTimer = setTimeout(showNext, 125);
  };
  if (animate && !matchMedia("(prefers-reduced-motion: reduce)").matches) showNext();
  else cy.batch(() => {
    for (const item of path) item.removeClass("path-dim").addClass("path-member");
  });
}

function updatePathButtons() {
  $("clear-path").hidden = !activeQuestion;
  const entries = questionReport?.questions.filter((entry) => entry.start_type) ?? [];
  $("question-results").querySelectorAll("button.question-item").forEach((row, index) => {
    row.setAttribute("aria-pressed", entries[index] === activeQuestion);
  });
}

function clearPath() {
  clearTimeout(pathTimer);
  activeQuestion = null;
  cy.elements().removeClass("path-member path-dim");
  updatePathButtons();
}

function togglePath(entry) {
  if (activeQuestion === entry) {
    clearPath();
    return;
  }
  if (yamlOpen) showYaml(false);
  if (!pathElements(entry).length) return;
  activeQuestion = entry;
  paintPath(true);
  updatePathButtons();
}

async function runJob(kind, payload, onDone) {
  if (activeJob) return;
  activeJob = { kind, started: performance.now() };
  updateControls();
  const timer = setInterval(updateControls, 1000);
  try {
    const { job_id: id } = await api(`jobs/${kind}`, payload);
    while (true) {
      await new Promise((resolve) => setTimeout(resolve, 1000));
      const job = await api(`jobs/${id}`);
      if (job.status === "running") continue;
      if (job.status === "error") throw new Error(job.error);
      await onDone(job.result);
      break;
    }
  } finally {
    clearInterval(timer);
    activeJob = null;
    updateControls();
  }
}

function setQuestionReport(report) {
  clearPath();
  questionReport = report;
  renderQuestions();
  if (report) showTab("questions");
}

async function applySuggestion(result) {
  if (dirty && !await askApproval(
    "Replace unsaved changes?", "This suggestion will replace the current ontology.", "Replace",
  )) return;
  clearTimeout(validationTimer);
  revision += 1;
  yamlDraft = null;
  yamlText = result.yaml;
  validation = { revision, promise: Promise.resolve(true) };
  $("yaml-editor").value = yamlText;
  updateProblems([]);
  setDirty(true);
  replaceOntology(result.ontology);
  setQuestionReport(result.coverage);
  toast("Ontology suggestion ready");
}

function applyPreview(report, structure) {
  previewReport = report;
  previewOutdated = structure !== structureOf(ontology);
  $("workbench-status").textContent = previewOutdated ? "Preview is out of date" : "";
  renderPreview();
  renderInspectorPreview();
  paintGraph();
  showTab("preview");
}

const dialogs = {
  suggest: $("suggest-dialog"),
  preview: $("preview-dialog"),
  "check-questions": $("check-dialog"),
};

function openJobDialog(kind) {
  const dialog = dialogs[kind];
  const form = dialog.querySelector("form");
  const fields = form.elements;
  if (kind === "suggest") {
    $("extend-option").hidden = !ontology;
    if (!ontology) fields.extend.checked = false;
    fields.useQuestions.disabled = !questions().length;
  }
  if (kind === "check-questions") {
    $("check-count").textContent = plural(questions().length, "question");
  }
  dialog.showModal();
}

function handleForm(formId, submit) {
  const form = $(formId);
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const data = new FormData(form);
    form.closest("dialog").close();
    submit(data).catch(showError);
  });
}

handleForm("suggest-form", async (data) => {
  const extend = data.has("extend");
  const baseYaml = extend ? await currentYaml() : null;
  if (extend && baseYaml === null) return;
  await runJob("suggest", {
    questions: data.has("useQuestions") ? questions() : [],
    instructions: data.get("instructions").trim() || null,
    base_yaml: baseYaml,
    repair: data.has("repair"),
    hops: Number(data.get("hops")),
  }, applySuggestion);
});

handleForm("preview-form", async (data) => {
  const yaml = await currentYaml();
  if (yaml === null) return;
  const structure = structureOf(ontology);
  await runJob("preview", {
    yaml,
    documents: Number(data.get("documents")),
    chunks_per_doc: Number(data.get("chunks")),
  }, (report) => applyPreview(report, structure));
});

handleForm("check-form", async (data) => {
  const yaml = await currentYaml();
  if (yaml === null) return;
  await runJob("check-questions", {
    yaml, questions: questions(), hops: Number(data.get("hops")),
  }, setQuestionReport);
});

handleForm("save-form", async (data) => {
  const yaml = await currentYaml();
  if (yaml === null) return;
  const submittedRevision = revision;
  const path = data.get("path").trim() || null;
  saving = true;
  updateControls();
  try {
    let result;
    try {
      result = await api("ontology/save", { yaml, path, overwrite: false });
    } catch (error) {
      if (!(error instanceof ApiError) || error.status !== 409 || !error.payload.exists) throw error;
      const target = path ?? workspace.ontology_path ?? "ontology.yaml";
      if (!await askApproval("Overwrite ontology?", `Replace the contents of ${target}?`, "Overwrite")) return;
      result = await api("ontology/save", { yaml, path, overwrite: true });
    }
    setPath(result.path);
    if (revision === submittedRevision) setDirty(false);
    toast(`Saved ${result.path}`);
  } finally {
    saving = false;
    updateControls();
  }
});

$("save").addEventListener("click", () => {
  $("save-form").elements.path.value = workspace.ontology_path ?? "ontology.yaml";
  $("save-dialog").showModal();
});
for (const node of jobButtons) node.addEventListener("click", () => openJobDialog(node.dataset.job));
for (const cancel of document.querySelectorAll("[data-cancel]")) {
  cancel.addEventListener("click", () => cancel.closest("dialog").close());
}

$("add-entity").addEventListener("click", addEntity);
$("start-empty").addEventListener("click", addEntity);
$("add-relation").addEventListener("click", addRelation);
$("layout").addEventListener("click", rerunLayout);
$("zoom-out").addEventListener("click", () => zoomGraph(Math.max(cy.minZoom(), cy.zoom() / 1.25), true));
$("zoom-in").addEventListener("click", () => zoomGraph(Math.min(cy.maxZoom(), cy.zoom() * 1.25), true));
$("fit").addEventListener("click", fitGraph);
$("graph").addEventListener("keydown", (event) => {
  const offsets = { ArrowLeft: [40, 0], ArrowRight: [-40, 0], ArrowUp: [0, 40], ArrowDown: [0, -40] };
  const offset = offsets[event.key];
  if (!offset) return;
  event.preventDefault();
  viewportMoved = true;
  cy.panBy({ x: offset[0], y: offset[1] });
});
$("type-picker").addEventListener("change", (event) => {
  const id = event.target.value;
  for (const [kind, items] of [["entity", ontology?.entities], ["relation", ontology?.relations]]) {
    const item = items?.find((entry) => graphId(entry) === id);
    if (item) {
      selectType({ kind, item });
      return;
    }
  }
  selectType(null);
});
$("questions").addEventListener("input", updateControls);
$("show-yaml").addEventListener("click", () => showYaml(!yamlOpen));
$("yaml-editor").addEventListener("input", (event) => {
  yamlDraft = event.target.value;
  scheduleValidation();
});
$("dismiss-banner").addEventListener("click", () => { $("banner").hidden = true; });
$("clear-path").addEventListener("click", clearPath);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !document.querySelector("dialog[open]")) clearPath();
});
window.addEventListener("beforeunload", (event) => {
  if (!dirty) return;
  event.preventDefault();
  event.returnValue = "";
});
for (const [index, tab] of tabs.entries()) {
  tab.addEventListener("click", () => showTab(tab.id.slice(4)));
  tab.addEventListener("keydown", (event) => {
    const targetIndex = { ArrowLeft: (index + tabs.length - 1) % tabs.length,
      ArrowRight: (index + 1) % tabs.length, Home: 0, End: tabs.length - 1 }[event.key];
    if (targetIndex === undefined) return;
    event.preventDefault();
    showTab(tabs[targetIndex].id.slice(4), true);
  });
}

async function initialize() {
  setupReportResize();
  updateSchema();
  renderPreview();
  if (!token) throw new Error(openLinkMessage);
  workspace = await api("state");
  $("document-count").textContent = plural(workspace.document_count, "document");
  $("document-count").title = workspace.documents.join("\n");
  const skipped = workspace.skipped_documents;
  if (skipped) {
    const label = `${plural(skipped, "document")} skipped`;
    $("document-count").textContent += ` · ${label}`;
    $("document-count").title += `\n${label} because the path could not be resolved inside the workspace.`;
  }
  $("model").textContent = workspace.model ? `Model · ${workspace.model}` : workspace.model_error;
  $("model").classList.toggle("warning", !workspace.model);
  if (workspace.model_error) $("model").title = workspace.model_error;
  setPath(workspace.ontology_path);
  $("questions").value = workspace.questions.join("\n");
  const defaults = workspace.defaults;
  $("suggest-form").elements.hops.value = defaults.hops;
  $("preview-form").elements.documents.value = defaults.documents;
  $("preview-form").elements.chunks.value = defaults.chunks_per_doc;
  $("check-form").elements.hops.value = defaults.hops;
  yamlDraft = workspace.ontology_yaml;
  updateSchema();
  const initialRevision = revision;
  if (yamlDraft !== null && !(await validate()) && revision === initialRevision) showYaml(true);
}

initialize().catch((error) => {
  $("empty-title").textContent = "Workspace unavailable";
  showError(error);
});
