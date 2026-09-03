import { Badge, EmptyState } from "../shared/primitives";
import type { ResultTabDefinition, SurfaceResponse } from "./types";

interface TimelinePoint {
  unitId: string;
  sequenceIndex: number;
  clusterId: string;
  semanticShift: number;
  clusterActivation: number;
}

interface SemanticCluster {
  id: string;
  representativeText: string;
  memberUnitIds: string[];
  meanSimilarity: number;
}

interface SemanticHotspot {
  clusterId: string;
  coverage: number;
  persistence: number;
  meanActivation: number;
  peakSequenceIndex: number;
}

interface GraphNode {
  id: string;
  kind: string;
  label: string;
  sequenceIndex?: number;
  confidence?: number;
}

interface GraphEdge {
  sourceId: string;
  targetId: string;
  kind: string;
  label?: string;
  weight?: number;
}

interface PositionedGraphNode extends GraphNode {
  x: number;
  y: number;
  group: GraphNodeGroup;
}

type GraphNodeGroup = "concept" | "unit" | "linguistic";

const GRAPH_WIDTH = 820;
const GRAPH_COLUMN_X: Record<GraphNodeGroup, number> = {
  concept: 110,
  unit: 410,
  linguistic: 710,
};

export function createSemanticMapResultTabs(
  operation = "analysis.semantic-map",
): ResultTabDefinition[] {
  return [
    {
      id: "semantic-map",
      label: "Map",
      render: (response) => <SemanticMapPanel operation={operation} response={response} />,
    },
  ];
}

export function SemanticMapPanel({
  operation = "analysis.semantic-map",
  response,
}: {
  operation?: string;
  response: SurfaceResponse | null;
}) {
  if (!response || response.operation !== operation) {
    return (
      <div className="mt-4">
        <EmptyState>Run the semantic-map operation to explore its trajectory and graph.</EmptyState>
      </div>
    );
  }

  const payload = semanticPayload(response.value);
  const semantic = asRecord(payload.semantic);
  const timeline = timelinePoints(semantic.timeline);
  const clusters = semanticClusters(semantic.clusters);
  const hotspots = semanticHotspots(semantic.hotspots);
  const graph = asRecord(payload.linguisticGraph);
  const nodes = graphNodes(graph.nodes);
  const edges = graphEdges(graph.edges);

  if (timeline.length === 0 && clusters.length === 0) {
    return (
      <div className="mt-4">
        <EmptyState>The semantic-map response did not contain timeline or concept data.</EmptyState>
      </div>
    );
  }

  return (
    <div className="mt-4 space-y-5">
      <section className="rounded-md border border-zinc-200 bg-zinc-50 p-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-teal-700">Semantic map</p>
            <h2 className="mt-1 text-lg font-semibold text-zinc-950">Meaning over sequence</h2>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-zinc-600">
              Follow concept membership and semantic shifts through the source order, then inspect how linguistic evidence attaches to the same units.
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <Badge>{timeline.length} timeline points</Badge>
            <Badge>{clusters.length} concepts</Badge>
            <Badge>{nodes.length} graph nodes</Badge>
          </div>
        </div>
      </section>

      <SemanticTimeline timeline={timeline} clusters={clusters} />
      <HotspotPanel hotspots={hotspots} clusters={clusters} />
      <SemanticGraph nodes={nodes} edges={edges} />
    </div>
  );
}

function SemanticTimeline({
  timeline,
  clusters,
}: {
  timeline: TimelinePoint[];
  clusters: SemanticCluster[];
}) {
  if (timeline.length === 0) {
    return <SectionEmpty title="Semantic trajectory">No timeline points were returned.</SectionEmpty>;
  }

  const clusterIds = unique(timeline.map((point) => point.clusterId));
  const laneByCluster = new Map(clusterIds.map((clusterId, index) => [clusterId, index]));
  const maxSequence = Math.max(1, ...timeline.map((point) => point.sequenceIndex));
  const width = 820;
  const left = 116;
  const right = 28;
  const laneHeight = 54;
  const top = 34;
  const height = Math.max(150, top + clusterIds.length * laneHeight + 38);
  const xFor = (sequenceIndex: number) => left + (sequenceIndex / maxSequence) * (width - left - right);
  const yFor = (clusterId: string) => top + (laneByCluster.get(clusterId) ?? 0) * laneHeight;
  const ordered = [...timeline].sort((leftPoint, rightPoint) => leftPoint.sequenceIndex - rightPoint.sequenceIndex);
  const points = ordered.map((point) => `${xFor(point.sequenceIndex)},${yFor(point.clusterId)}`).join(" ");
  const representativeByCluster = new Map(clusters.map((cluster) => [cluster.id, cluster.representativeText]));

  return (
    <section className="overflow-hidden rounded-md border border-zinc-200 bg-white">
      <SectionHeader title="Semantic trajectory" detail="Sequence position × concept lane; larger markers indicate stronger semantic shifts." />
      <div className="overflow-x-auto p-4">
        <svg
          aria-label="Semantic trajectory chart"
          className="min-w-[720px] text-zinc-300"
          role="img"
          viewBox={`0 0 ${width} ${height}`}
        >
          {clusterIds.map((clusterId) => {
            const y = yFor(clusterId);
            return (
              <g key={clusterId}>
                <line className="stroke-zinc-200" x1={left} x2={width - right} y1={y} y2={y} />
                <text className="fill-zinc-600 text-[11px]" textAnchor="end" x={left - 12} y={y + 4}>
                  {clusterId}
                </text>
              </g>
            );
          })}
          {ordered.length > 1 ? <polyline className="fill-none stroke-teal-300" points={points} strokeWidth="2" /> : null}
          {ordered.map((point) => {
            const x = xFor(point.sequenceIndex);
            const y = yFor(point.clusterId);
            const radius = 5 + clamp01(point.semanticShift) * 7;
            return (
              <g key={point.unitId} className="text-teal-700">
                <circle className="fill-current stroke-white" cx={x} cy={y} r={radius} strokeWidth="2">
                  <title>
                    {`${point.unitId} · ${point.clusterId} · shift ${formatScore(point.semanticShift)} · activation ${formatScore(point.clusterActivation)} · ${representativeByCluster.get(point.clusterId) ?? ""}`}
                  </title>
                </circle>
                <text className="fill-zinc-500 text-[10px]" textAnchor="middle" x={x} y={y + radius + 16}>
                  {point.sequenceIndex}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
    </section>
  );
}

function HotspotPanel({
  hotspots,
  clusters,
}: {
  hotspots: SemanticHotspot[];
  clusters: SemanticCluster[];
}) {
  if (hotspots.length === 0) {
    return null;
  }
  const clusterById = new Map(clusters.map((cluster) => [cluster.id, cluster]));

  return (
    <section className="overflow-hidden rounded-md border border-zinc-200 bg-white">
      <SectionHeader title="Concept hotspots" detail="Coverage shows how much of the primary sequence belongs to a concept." />
      <div className="divide-y divide-zinc-100">
        {hotspots.slice(0, 8).map((hotspot) => {
          const cluster = clusterById.get(hotspot.clusterId);
          return (
            <div className="grid gap-3 px-4 py-3 lg:grid-cols-[13rem_minmax(0,1fr)_10rem]" key={hotspot.clusterId}>
              <div className="min-w-0">
                <p className="text-sm font-semibold text-zinc-950">{hotspot.clusterId}</p>
                <p className="mt-1 truncate text-xs text-zinc-500" title={cluster?.representativeText}>
                  {cluster?.representativeText ?? "No representative text"}
                </p>
              </div>
              <div className="flex items-center">
                <div className="h-2 w-full overflow-hidden rounded-full bg-zinc-100">
                  <div className="h-full rounded-full bg-teal-600" style={{ width: `${clamp01(hotspot.coverage) * 100}%` }} />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-2 text-xs tabular-nums text-zinc-600">
                <span>{Math.round(clamp01(hotspot.coverage) * 100)}% coverage</span>
                <span>peak {hotspot.peakSequenceIndex}</span>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function SemanticGraph({ nodes, edges }: { nodes: GraphNode[]; edges: GraphEdge[] }) {
  if (nodes.length === 0) {
    return <SectionEmpty title="Concept graph">Enable linguistic graph projection to inspect typed semantic evidence.</SectionEmpty>;
  }

  const positioned = positionGraphNodes(nodes);
  const byId = new Map(positioned.map((node) => [node.id, node]));
  const visibleEdges = edges
    .filter((edge) => byId.has(edge.sourceId) && byId.has(edge.targetId))
    .slice(0, 72);
  const maxY = Math.max(220, ...positioned.map((node) => node.y + 42));

  return (
    <section className="overflow-hidden rounded-md border border-zinc-200 bg-white">
      <SectionHeader
        title="Concept graph"
        detail="Concepts and source units stay central; linguistic entities, events, relations, discourse and topics are projected onto them."
      />
      <div className="flex flex-wrap gap-2 border-b border-zinc-100 px-4 py-3">
        <Badge>{positioned.filter((node) => node.group === "concept").length} concepts</Badge>
        <Badge>{positioned.filter((node) => node.group === "unit").length} units</Badge>
        <Badge>{positioned.filter((node) => node.group === "linguistic").length} linguistic nodes</Badge>
        <Badge>{visibleEdges.length} visible edges</Badge>
      </div>
      <div className="overflow-auto p-4">
        <svg
          aria-label="Semantic linguistic graph"
          className="min-w-[760px]"
          role="img"
          viewBox={`0 0 ${GRAPH_WIDTH} ${maxY}`}
        >
          {visibleEdges.map((edge, index) => {
            const source = byId.get(edge.sourceId);
            const target = byId.get(edge.targetId);
            if (!source || !target) return null;
            return (
              <line
                className={edge.kind === "semanticNeighbor" ? "stroke-teal-300" : "stroke-zinc-200"}
                key={`${edge.sourceId}-${edge.targetId}-${edge.kind}-${index}`}
                strokeWidth={edge.kind === "semanticNeighbor" ? 2 : 1.2}
                x1={source.x}
                x2={target.x}
                y1={source.y}
                y2={target.y}
              >
                <title>{`${edge.kind}${edge.label ? ` · ${edge.label}` : ""}${edge.weight == null ? "" : ` · ${formatScore(edge.weight)}`}`}</title>
              </line>
            );
          })}
          {positioned.map((node) => (
            <GraphNodeGlyph key={node.id} node={node} />
          ))}
          <text className="fill-zinc-500 text-[11px] font-semibold uppercase" textAnchor="middle" x={GRAPH_COLUMN_X.concept} y={18}>
            Concepts
          </text>
          <text className="fill-zinc-500 text-[11px] font-semibold uppercase" textAnchor="middle" x={GRAPH_COLUMN_X.unit} y={18}>
            Source units
          </text>
          <text className="fill-zinc-500 text-[11px] font-semibold uppercase" textAnchor="middle" x={GRAPH_COLUMN_X.linguistic} y={18}>
            Linguistic evidence
          </text>
        </svg>
      </div>
      {nodes.length > positioned.length || edges.length > visibleEdges.length ? (
        <p className="border-t border-zinc-100 px-4 py-3 text-xs text-zinc-500">
          The visualization shows a deterministic subset for readability. Open JSON for the complete graph.
        </p>
      ) : null}
    </section>
  );
}

function GraphNodeGlyph({ node }: { node: PositionedGraphNode }) {
  const className =
    node.group === "concept"
      ? "fill-teal-50 stroke-teal-400"
      : node.group === "unit"
        ? "fill-sky-50 stroke-sky-300"
        : "fill-zinc-50 stroke-zinc-300";
  return (
    <g>
      <rect className={className} height="30" rx="5" width="170" x={node.x - 85} y={node.y - 15}>
        <title>{`${node.kind} · ${node.label}${node.confidence == null ? "" : ` · confidence ${formatScore(node.confidence)}`}`}</title>
      </rect>
      <text className="fill-zinc-800 text-[10px]" textAnchor="middle" x={node.x} y={node.y + 3}>
        {truncate(node.label, 24)}
      </text>
    </g>
  );
}

function positionGraphNodes(nodes: GraphNode[]): PositionedGraphNode[] {
  const grouped: Record<GraphNodeGroup, GraphNode[]> = {
    concept: nodes.filter((node) => node.kind === "concept").slice(0, 10),
    unit: nodes
      .filter((node) => node.kind === "unit")
      .sort((left, right) => (left.sequenceIndex ?? 0) - (right.sequenceIndex ?? 0))
      .slice(0, 14),
    linguistic: nodes
      .filter((node) => node.kind !== "concept" && node.kind !== "unit")
      .sort((left, right) => (left.sequenceIndex ?? Number.MAX_SAFE_INTEGER) - (right.sequenceIndex ?? Number.MAX_SAFE_INTEGER) || left.id.localeCompare(right.id))
      .slice(0, 20),
  };

  return (Object.keys(grouped) as GraphNodeGroup[]).flatMap((group) =>
    grouped[group].map((node, index) => ({
      ...node,
      group,
      x: GRAPH_COLUMN_X[group],
      y: 48 + index * 42,
    })),
  );
}

function SectionHeader({ title, detail }: { title: string; detail: string }) {
  return (
    <div className="border-b border-zinc-200 px-4 py-3">
      <h3 className="text-sm font-semibold text-zinc-950">{title}</h3>
      <p className="mt-1 text-xs leading-5 text-zinc-500">{detail}</p>
    </div>
  );
}

function SectionEmpty({ title, children }: { title: string; children: string }) {
  return (
    <section className="rounded-md border border-zinc-200 bg-white p-4">
      <h3 className="text-sm font-semibold text-zinc-950">{title}</h3>
      <p className="mt-2 text-sm text-zinc-500">{children}</p>
    </section>
  );
}

function semanticPayload(value: unknown): Record<string, unknown> {
  const root = asRecord(value);
  const first = asRecord(root.result);
  const second = asRecord(first.result);
  return [root, first, second].find((candidate) => candidate.semantic !== undefined) ?? root;
}

function timelinePoints(value: unknown): TimelinePoint[] {
  return recordArray(value).flatMap((entry) => {
    const unitId = stringValue(entry.unitId);
    const clusterId = stringValue(entry.clusterId);
    const sequenceIndex = numberValue(entry.sequenceIndex);
    if (!unitId || !clusterId || sequenceIndex == null) return [];
    return [{
      unitId,
      sequenceIndex,
      clusterId,
      semanticShift: numberValue(entry.semanticShift) ?? 0,
      clusterActivation: numberValue(entry.clusterActivation) ?? 0,
    }];
  });
}

function semanticClusters(value: unknown): SemanticCluster[] {
  return recordArray(value).flatMap((entry) => {
    const id = stringValue(entry.id);
    if (!id) return [];
    return [{
      id,
      representativeText: stringValue(entry.representativeText) ?? id,
      memberUnitIds: stringArray(entry.memberUnitIds),
      meanSimilarity: numberValue(entry.meanSimilarity) ?? 0,
    }];
  });
}

function semanticHotspots(value: unknown): SemanticHotspot[] {
  return recordArray(value).flatMap((entry) => {
    const clusterId = stringValue(entry.clusterId);
    if (!clusterId) return [];
    return [{
      clusterId,
      coverage: numberValue(entry.coverage) ?? 0,
      persistence: numberValue(entry.persistence) ?? 0,
      meanActivation: numberValue(entry.meanActivation) ?? 0,
      peakSequenceIndex: numberValue(entry.peakSequenceIndex) ?? 0,
    }];
  });
}

function graphNodes(value: unknown): GraphNode[] {
  return recordArray(value).flatMap((entry) => {
    const id = stringValue(entry.id);
    const kind = stringValue(entry.kind);
    if (!id || !kind) return [];
    return [{
      id,
      kind,
      label: stringValue(entry.label) ?? id,
      sequenceIndex: numberValue(entry.sequenceIndex),
      confidence: numberValue(entry.confidence),
    }];
  });
}

function graphEdges(value: unknown): GraphEdge[] {
  return recordArray(value).flatMap((entry) => {
    const sourceId = stringValue(entry.sourceId);
    const targetId = stringValue(entry.targetId);
    const kind = stringValue(entry.kind);
    if (!sourceId || !targetId || !kind) return [];
    return [{
      sourceId,
      targetId,
      kind,
      label: stringValue(entry.label),
      weight: numberValue(entry.weight),
    }];
  });
}

function recordArray(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value) ? value.map(asRecord).filter((entry) => Object.keys(entry).length > 0) : [];
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
}

function stringValue(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

function numberValue(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((entry): entry is string => typeof entry === "string") : [];
}

function unique(values: string[]): string[] {
  return [...new Set(values)];
}

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}

function formatScore(value: number): string {
  return value.toFixed(3);
}

function truncate(value: string, length: number): string {
  return value.length <= length ? value : `${value.slice(0, Math.max(0, length - 1))}…`;
}
