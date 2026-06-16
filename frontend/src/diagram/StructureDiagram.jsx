import { useMemo } from "react";
import { layoutDiagram } from "./layout";

function StructureDiagram({
  structure,
  zoom,
  selectedPath,
  matchedPaths,
  expandedGroups,
  searchActive,
  onSelectNode,
  onToggleGroup,
  showGroupToggle = true,
}) {
  const nodes = useMemo(
    () => layoutDiagram(structure.root, expandedGroups),
    [structure, expandedGroups]
  );
  const width = Math.max(960, ...nodes.map((node) => node.x + node.width + 40));
  const height = Math.max(430, ...nodes.map((node) => node.y + node.height + 40));
  const matched = matchedPaths instanceof Set ? matchedPaths : new Set();
  return (
    <div className="diagram-scroll">
      <div
        className="diagram-zoom"
        style={{ transform: `scale(${zoom})`, width, height }}
      >
        <svg
          className="diagram-svg"
          viewBox={`0 0 ${width} ${height}`}
          width={width}
          height={height}
          role="img"
          aria-label="Model architecture diagram"
        >
          <defs>
            <marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
              <path d="M0,0 L8,4 L0,8 Z" fill="var(--diagram-arrow)" />
            </marker>
          </defs>
          {nodes.flatMap((node) =>
            node.children.map((child) => {
              const target = nodes.find((item) => item.path === child);
              if (!target) return null;
              return (
                <path
                  key={`${node.path}-${child}`}
                  d={`M ${node.x + node.width} ${node.y + node.height / 2} C ${node.x + node.width + 36} ${
                    node.y + node.height / 2
                  }, ${target.x - 36} ${target.y + target.height / 2}, ${target.x} ${target.y + target.height / 2}`}
                  fill="none"
                  stroke="var(--diagram-arrow)"
                  strokeWidth="1.5"
                  markerEnd="url(#arrow)"
                />
              );
            })
          )}
          {nodes.map((node) => {
            const isSelected = selectedPath === node.path;
            const isMatch = matched.has(node.path);
            const isDimmed = searchActive && !isMatch;
            const classes = [
              "diagram-node",
              node.typeClass,
              isSelected ? "selected" : "",
              isMatch ? "match" : "",
              isDimmed ? "dimmed" : "",
            ]
              .filter(Boolean)
              .join(" ");
            return (
              <g
                key={node.path}
                transform={`translate(${node.x}, ${node.y})`}
                style={{ cursor: "pointer" }}
                onClick={() => onSelectNode && onSelectNode(node.path)}
              >
                <rect width={node.width} height={node.height} rx="10" className={classes} />
                <foreignObject x="0" y="0" width={node.width} height={node.height}>
                  <div xmlns="http://www.w3.org/1999/xhtml" className="diagram-node-content">
                    <div className="diagram-node-header">
                      <span className="diagram-title" title={node.fullName}>
                        {node.displayName}
                      </span>
                      {node.repeat && (
                        <span className="diagram-repeat">×{node.repeat}</span>
                      )}
                      {showGroupToggle && node.isLayerGroup && (
                        <button
                          className="layer-group-toggle"
                          onClick={(e) => {
                            e.stopPropagation();
                            onToggleGroup && onToggleGroup(node.path);
                          }}
                          title={node.isExpanded ? "Collapse layers" : "Expand layers"}
                        >
                          {node.isExpanded ? "−" : "+"}
                        </button>
                      )}
                    </div>
                    {node.metaLines.length > 0 && (
                      <ul className="diagram-meta">
                        {node.metaLines.map((line) => (
                          <li key={line} title={line}>
                            {line}
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                </foreignObject>
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );
}

export default StructureDiagram;
