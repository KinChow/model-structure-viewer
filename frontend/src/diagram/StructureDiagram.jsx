import { useEffect, useMemo, useRef, useState } from "react";
import { layoutDiagram } from "./layout";
import { fitDiagramViewport, sameDiagramViewport } from "./viewport";

function StructureDiagram({
  structure,
  zoom,
  fitNonce = 0,
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
  const contentWidth = Math.max(1, ...nodes.map((node) => node.x + node.width + 28));
  const contentHeight = Math.max(1, ...nodes.map((node) => node.y + node.height + 28));
  const frameRef = useRef(null);
  const scrollRef = useRef(null);
  const fitNonceRef = useRef(fitNonce);
  const [viewport, setViewport] = useState(() =>
    fitDiagramViewport({
      contentWidth,
      contentHeight,
      viewportWidth: 960,
      viewportHeight: 470,
      zoom,
    })
  );
  const matched = matchedPaths instanceof Set ? matchedPaths : new Set();

  useEffect(() => {
    const element = frameRef.current;
    if (!element) return;
    const update = () => {
      const nextViewport = fitDiagramViewport({
          contentWidth,
          contentHeight,
          viewportWidth: element.clientWidth,
          viewportHeight: element.clientHeight,
          zoom,
        });
      const shouldResetScroll = fitNonceRef.current !== fitNonce;
      setViewport((prev) => (sameDiagramViewport(prev, nextViewport) ? prev : nextViewport));
      if (shouldResetScroll && scrollRef.current) {
        scrollRef.current.scrollLeft = 0;
        scrollRef.current.scrollTop = 0;
        fitNonceRef.current = fitNonce;
      }
    };
    update();
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(update);
    observer.observe(element);
    return () => observer.disconnect();
  }, [contentWidth, contentHeight, fitNonce, zoom]);

  const width = viewport.canvasWidth;
  const height = viewport.canvasHeight;
  const contentTransform = `translate(${viewport.offsetX}, ${viewport.offsetY}) scale(${viewport.scale})`;

  return (
    <div className="diagram-frame" ref={frameRef}>
      <div className="diagram-scroll" ref={scrollRef}>
        <div className="diagram-zoom" style={{ width, height }}>
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
            <g transform={contentTransform}>
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
                          {showGroupToggle && node.isCollapsible && (
                            <button
                              className="layer-group-toggle"
                              onClick={(e) => {
                                e.stopPropagation();
                                onToggleGroup && onToggleGroup(node.path);
                              }}
                              title={node.isExpanded ? "Collapse" : "Expand"}
                              aria-label={node.isExpanded ? "Collapse" : "Expand"}
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
            </g>
          </svg>
        </div>
      </div>
    </div>
  );
}

export default StructureDiagram;
