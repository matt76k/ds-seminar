// 日本語の索引ページを作るディレクティブ `japanese-index`。
// mystmd 標準の show-index は英字の頭文字でしか分けられず、「よみ - 表記」の読みもそのまま表示するため、
// 原稿の {index}`表示<よみ - 表記>` を読み、表記を五十音の行ごとに並べる。
import { readFileSync } from "node:fs";

const INDEX_ROLE = /\{index\}`([^`]+)`/g;
const READING_KEY = /^([ぁ-ゖァ-ヺー]+)\s*-\s*(.+)$/;
const ROWS = [
  ["あ行", "あいうえお"], ["か行", "かきくけこ"], ["さ行", "さしすせそ"], ["た行", "たちつてと"], ["な行", "なにぬねの"],
  ["は行", "はひふへほ"], ["ま行", "まみむめも"], ["や行", "やゆよ"], ["ら行", "らりるれろ"], ["わ行", "わをん"],
];
const SMALL_KANA = { ぁ: "あ", ぃ: "い", ぅ: "う", ぇ: "え", ぉ: "お", っ: "つ", ゃ: "や", ゅ: "ゆ", ょ: "よ", ゎ: "わ" };
const ALPHANUMERIC = "英数字";
const OTHER = "その他";
const GROUP_ORDER = [ALPHANUMERIC, ...ROWS.map(([name]) => name), OTHER];

const toHiragana = (s) => s.replace(/[ァ-ヶ]/g, (c) => String.fromCharCode(c.charCodeAt(0) - 0x60));

function parseEntry(content) {
  const key = (content.match(/<(.+)>$/s)?.[1] ?? content).trim();
  const reading = key.match(READING_KEY);
  const display = reading ? reading[2].trim() : key;
  return { display, sortKey: toHiragana(reading ? reading[1] : key).toLowerCase() };
}

function groupOf(sortKey) {
  const first = sortKey[0];
  if (/[a-z0-9]/.test(first)) return ALPHANUMERIC;
  const base = SMALL_KANA[first] ?? first.normalize("NFD")[0];
  return ROWS.find(([, kana]) => kana.includes(base))?.[0] ?? OTHER;
}

function collectEntries() {
  const files = [...readFileSync("myst.yml", "utf8").matchAll(/- file: (\S+\.md)/g)].map((m) => m[1]);
  const entries = new Map();
  for (const file of files) {
    for (const [, content] of readFileSync(file, "utf8").matchAll(INDEX_ROLE)) {
      const { display, sortKey } = parseEntry(content);
      const entry = entries.get(display) ?? { display, sortKey, files: [] };
      if (!entry.files.includes(file)) entry.files.push(file);
      entries.set(display, entry);
    }
  }
  return [...entries.values()];
}

function entryItem({ display, files }) {
  const links = files.flatMap((file, i) => [
    ...(i ? [{ type: "text", value: ", " }] : []),
    { type: "link", url: `/${file}`, children: [] },
  ]);
  return { type: "listItem", spread: false, children: [{ type: "paragraph", children: [{ type: "text", value: `${display}　` }, ...links] }] };
}

const japaneseIndexDirective = {
  name: "japanese-index",
  doc: "原稿の {index} ロールから五十音順の索引を作る",
  run() {
    const groups = Map.groupBy(collectEntries(), (entry) => groupOf(entry.sortKey));
    return GROUP_ORDER.filter((name) => groups.has(name)).flatMap((name) => [
      { type: "heading", depth: 2, enumerated: false, children: [{ type: "text", value: name }] },
      {
        type: "list",
        ordered: false,
        spread: false,
        children: groups.get(name).sort((a, b) => a.sortKey.localeCompare(b.sortKey, "ja")).map(entryItem),
      },
    ]);
  },
};

export default { name: "Japanese index", directives: [japaneseIndexDirective] };
