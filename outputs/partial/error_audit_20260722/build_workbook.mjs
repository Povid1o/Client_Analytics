import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const baseDir = "/Users/nikitabaslykov/Documents/Python/ML/Client_Analytics/outputs/error_audit_20260722";
const readJson = async (name) => JSON.parse(await fs.readFile(`${baseDir}/${name}`, "utf8"));
const overall = await readJson("overall_metrics.json");
const groups = await readJson("group_metrics.json");
const confusion = await readJson("confusion.json");
const samples = await readJson("samples.json");
const worstCases = await readJson("worst_cases.json");

const workbook = Workbook.create();
const navy = "#17365D";
const blue = "#2F75B5";
const lightBlue = "#D9EAF7";
const paleBlue = "#EDF4FA";
const green = "#E2F0D9";
const amber = "#FFF2CC";
const red = "#FCE4D6";
const gray = "#E7E6E6";
const white = "#FFFFFF";
const bandOrder = ["20–50k", "50–75k", "75–100k", "100–150k", "150–250k", "250–400k", "400–700k", "700k+"];

function colName(index) {
  let value = index + 1;
  let result = "";
  while (value > 0) {
    const rem = (value - 1) % 26;
    result = String.fromCharCode(65 + rem) + result;
    value = Math.floor((value - 1) / 26);
  }
  return result;
}

function normalizeRows(records, columns) {
  return records.map((row) => columns.map((column) => row[column] ?? null));
}

function styleTitle(range) {
  range.format = {
    fill: navy,
    font: { bold: true, color: white, size: 16 },
    verticalAlignment: "center",
  };
  range.format.rowHeight = 30;
}

function setColumnWidths(sheet, columns) {
  columns.forEach((column, index) => {
    const range = sheet.getRange(`${colName(index)}:${colName(index)}`);
    let width = 14;
    if (["id", "rows", "true_band", "predicted_band", "band_distance", "anchor_count"].includes(column)) width = 10;
    if (column.includes("band_name") || column === "group" || column === "group_type") width = 19;
    if (["model", "diagnostic_flag", "source_quality", "route_error_bucket"].includes(column)) width = 24;
    if (["adminarea", "gender", "dt", "split"].includes(column)) width = 15;
    if (column.includes("prediction") || column.includes("correction") || column.includes("error") || column.includes("bias") || column.includes("WMAE")) width = 18;
    if (column.startsWith("dp_") || column === "salary_6to12m_avg" || column === "incomeValue") width = 20;
    range.format.columnWidth = width;
  });
}

function applySemanticFormats(sheet, columns, rowCount) {
  columns.forEach((column, index) => {
    const letter = colName(index);
    const dataRange = sheet.getRange(`${letter}2:${letter}${rowCount + 1}`);
    const lower = column.toLowerCase();
    if (lower === "w") {
      dataRange.format.numberFormat = "0.000";
    } else if (lower.includes("rate") || lower.includes("share") || lower.includes("accuracy") || lower.includes("within") || lower === "weighted_mape" || lower === "weighted_r2" || lower.startsWith("p_band_")) {
      dataRange.format.numberFormat = "0.0%";
    } else if (["confidence", "entropy", "posterior_spread", "calibration_slope_y_on_prediction"].includes(lower)) {
      dataRange.format.numberFormat = "0.000";
    } else if (["id", "rows", "band_distance", "anchor_count", "error_quintile_within_band"].includes(lower)) {
      dataRange.format.numberFormat = "#,##0";
    } else if (lower === "dt") {
      dataRange.format.numberFormat = "yyyy-mm-dd";
    } else if (!lower.includes("band") && !["split", "model", "group", "group_type", "gender", "adminarea", "source_quality", "diagnostic_flag", "salary_present"].includes(lower)) {
      dataRange.format.numberFormat = "#,##0;[Red]-#,##0";
    }
  });
}

function addDataSheet(name, records, tableName, freezeColumns = 2) {
  const sheet = workbook.worksheets.add(name);
  sheet.showGridLines = false;
  if (!records.length) return sheet;
  const columns = Object.keys(records[0]);
  const values = [columns, ...normalizeRows(records, columns)];
  const end = colName(columns.length - 1);
  sheet.getRange(`A1:${end}${values.length}`).values = values;
  const header = sheet.getRange(`A1:${end}1`);
  header.format = {
    fill: navy,
    font: { bold: true, color: white },
    wrapText: true,
    verticalAlignment: "center",
  };
  header.format.rowHeight = 44;
  const table = sheet.tables.add(`A1:${end}${values.length}`, true, tableName);
  table.style = "TableStyleMedium2";
  table.showFilterButton = true;
  sheet.freezePanes.freezeRows(1);
  if (freezeColumns) sheet.freezePanes.freezeColumns(freezeColumns);
  applySemanticFormats(sheet, columns, records.length);
  setColumnWidths(sheet, columns);
  const wmaeIndex = columns.findIndex((column) => column === "local_WMAE" || column === "WMAE");
  if (wmaeIndex >= 0) {
    sheet.getRange(`${colName(wmaeIndex)}2:${colName(wmaeIndex)}${records.length + 1}`).conditionalFormats.add("colorScale", {
      colors: ["#E2F0D9", "#FFF2CC", "#F4CCCC"], thresholds: ["min", "50%", "max"],
    });
  }
  const biasIndex = columns.findIndex((column) => column === "weighted_mean_bias_pred_minus_target" || column === "signed_error");
  if (biasIndex >= 0) {
    sheet.getRange(`${colName(biasIndex)}2:${colName(biasIndex)}${records.length + 1}`).conditionalFormats.add("colorScale", {
      colors: ["#9DC3E6", "#FFFFFF", "#F4B183"], thresholds: ["min", 0, "max"],
    });
  }
  const distanceIndex = columns.indexOf("band_distance");
  if (distanceIndex >= 0) {
    sheet.getRange(`${colName(distanceIndex)}2:${colName(distanceIndex)}${records.length + 1}`).conditionalFormats.add(
      "cellIs", { operator: "greaterThanOrEqual", formula: 2, format: { fill: "#F4CCCC", font: { color: "#9C0006", bold: true } } }
    );
  }
  return sheet;
}

const summary = workbook.worksheets.add("Резюме");
summary.showGridLines = false;
summary.mergeCells("A1:Q1");
summary.getRange("A1").values = [["Аудит ошибок модели дохода — OOF random и temporal"]];
styleTitle(summary.getRange("A1:Q1"));
summary.mergeCells("A2:Q2");
summary.getRange("A2").values = [["Bias = prediction − target: отрицательное значение означает недопрогноз. Метрики рассчитаны на лучшем фиксированном minimax-ансамбле."]];
summary.getRange("A2:Q2").format = { fill: paleBlue, font: { color: "#404040", italic: true }, wrapText: true };
summary.getRange("A2:Q2").format.rowHeight = 28;

const models = ["Base CatBoost", "Ordinal only", "Final minimax ensemble"];
summary.getRange("A4:C7").values = [
  ["Модель", "Random WMAE", "Temporal WMAE"],
  ...models.map((model) => [
    model,
    overall.find((row) => row.split === "random" && row.model === model).WMAE,
    overall.find((row) => row.split === "temporal" && row.model === model).WMAE,
  ]),
];
summary.getRange("A4:C4").format = { fill: blue, font: { bold: true, color: white } };
summary.getRange("B5:C7").format.numberFormat = "#,##0";
summary.getRange("A4:C7").format.borders = { preset: "outside", style: "thin", color: "#A6A6A6" };
summary.getRange("A:A").format.columnWidth = 25;
summary.getRange("B:C").format.columnWidth = 16;

const trueBands = groups
  .filter((row) => row.group_type === "true income band")
  .sort((a, b) => a.split.localeCompare(b.split) || bandOrder.indexOf(a.group) - bandOrder.indexOf(b.group));
summary.getRange("J4:L12").values = [
  ["Истинная группа", "Random local WMAE", "Temporal local WMAE"],
  ...bandOrder.map((band) => [
    band,
    trueBands.find((row) => row.split === "random" && row.group === band).local_WMAE,
    trueBands.find((row) => row.split === "temporal" && row.group === band).local_WMAE,
  ]),
];
summary.getRange("J4:L4").format = { fill: blue, font: { bold: true, color: white } };
summary.getRange("K5:L12").format.numberFormat = "#,##0";
summary.getRange("J:J").format.columnWidth = 18;
summary.getRange("K:L").format.columnWidth = 18;

summary.getRange("A9:H9").merge();
summary.getRange("A9").values = [["Ключевые выводы"]];
summary.getRange("A9:H9").format = { fill: navy, font: { bold: true, color: white } };
const findings = [
  "1. Основная поломка — far routing: промах на ≥2 диапазона занимает ~16% веса и создаёт 25.8–28.6k глобального WMAE.",
  "2. При 0–1 income anchors локальный WMAE 67.8–69.4k; при 3+ anchors — 41.4–41.8k.",
  "3. Диапазоны 50–150k не улучшаются стабильно; группа 75–100k распознаётся лишь примерно в 10–11% взвешенных случаев.",
  "4. Верхний хвост систематически недопрогнозируется: bias −107…−113k для 400–700k и −344…−350k для 700k+.",
];
findings.forEach((text, i) => {
  const row = 10 + i;
  summary.mergeCells(`A${row}:H${row}`);
  summary.getRange(`A${row}`).values = [[text]];
  summary.getRange(`A${row}:H${row}`).format = { fill: i % 2 ? "#F7F9FB" : white, wrapText: true };
  summary.getRange(`A${row}:H${row}`).format.rowHeight = 25;
});

const overallChart = summary.charts.add("bar", summary.getRange("A4:C7"));
overallChart.title = "WMAE: база против ансамбля";
overallChart.hasLegend = true;
overallChart.yAxis = { numberFormatCode: "#,##0" };
overallChart.setPosition("A15", "H30");

const bandChart = summary.charts.add("bar", summary.getRange("J4:L12"));
bandChart.title = "Local WMAE по истинной группе";
bandChart.hasLegend = true;
bandChart.yAxis = { numberFormatCode: "#,##0" };
bandChart.setPosition("J15", "Q30");
summary.freezePanes.freezeRows(2);

addDataSheet("Общие метрики", overall, "OverallMetrics", 2);
addDataSheet("Истинные группы", trueBands, "TrueBandMetrics", 3);
const predictedBands = groups
  .filter((row) => row.group_type === "predicted income band")
  .sort((a, b) => a.split.localeCompare(b.split) || bandOrder.indexOf(a.group) - bandOrder.indexOf(b.group));
addDataSheet("Предсказанные группы", predictedBands, "PredictedBandMetrics", 3);
const failureModes = groups.filter((row) => !["true income band", "predicted income band"].includes(row.group_type));
addDataSheet("Срезы ошибок", failureModes, "FailureModeMetrics", 3);

function addConfusionSheet(name, split, tableName) {
  const sheet = workbook.worksheets.add(name);
  sheet.showGridLines = false;
  const filtered = confusion.filter((row) => row.split === split);
  const matrix = [["Истинная \\ Предсказанная", ...bandOrder]];
  for (const trueBand of bandOrder) {
    matrix.push([
      trueBand,
      ...bandOrder.map((predictedBand) => filtered.find((row) => row.true_band === trueBand && row.predicted_band === predictedBand).weighted_row_share),
    ]);
  }
  sheet.getRange("A1:I9").values = matrix;
  sheet.getRange("A1:I1").format = { fill: navy, font: { bold: true, color: white }, wrapText: true };
  sheet.getRange("A2:A9").format = { fill: lightBlue, font: { bold: true } };
  sheet.getRange("B2:I9").format.numberFormat = "0.0%";
  sheet.getRange("B2:I9").conditionalFormats.add("colorScale", {
    colors: ["#FFFFFF", "#FFF2CC", "#63BE7B"], thresholds: ["min", "50%", "max"],
  });
  sheet.getRange("A:I").format.columnWidth = 15;
  sheet.getRange("A:A").format.columnWidth = 24;
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(1);
  return sheet;
}
addConfusionSheet("Confusion random", "random", "ConfusionRandom");
addConfusionSheet("Confusion temporal", "temporal", "ConfusionTemporal");

addDataSheet("200 random на группу", samples.filter((row) => row.split === "random"), "RandomSamples", 5);
addDataSheet("200 temporal на группу", samples.filter((row) => row.split === "temporal"), "TemporalSamples", 5);
addDataSheet("Худшие случаи", worstCases, "WorstCases", 5);

const inspect = await workbook.inspect({
  kind: "table",
  range: "Резюме!A1:Q30",
  include: "values,formulas",
  tableMaxRows: 30,
  tableMaxCols: 17,
});
await fs.writeFile(`${baseDir}/summary_inspect.ndjson`, inspect.ndjson ?? "", "utf8");
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
await fs.writeFile(`${baseDir}/formula_errors.ndjson`, errors.ndjson ?? "", "utf8");

const previewDir = `${baseDir}/previews`;
await fs.mkdir(previewDir, { recursive: true });
for (const sheetName of [
  "Резюме", "Общие метрики", "Истинные группы", "Предсказанные группы",
  "Срезы ошибок", "Confusion random", "Confusion temporal",
  "200 random на группу", "200 temporal на группу", "Худшие случаи",
]) {
  const preview = await workbook.render({
    sheetName,
    range: sheetName === "Резюме" ? "A1:Q30" : "A1:Q25",
    scale: 1,
    format: "png",
  });
  await fs.writeFile(`${previewDir}/${sheetName.replaceAll(" ", "_")}.png`, new Uint8Array(await preview.arrayBuffer()));
}

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(`${baseDir}/income_model_error_audit.xlsx`);
