import AttributeGrid from "./AttributeGrid";
import { typeClass } from "../diagram/meta";

function pathChild(path, index) {
  return `${path}.${index}`;
}

function ModuleCards({
  node,
  depth,
  path = "root",
  selectedPath,
  matchedPaths,
  expandedGroups,
  searchActive,
  onSelectNode,
  onToggleGroup,
}) {
  const matched = matchedPaths instanceof Set ? matchedPaths : new Set();
  const expanded = expandedGroups instanceof Set ? expandedGroups : new Set();
  const isSelected = selectedPath === path;
  const isMatch = matched.has(path);
  const isDimmed = searchActive && !isMatch;
  const isLayerGroup = node.type === "layer-group";
  const isExpanded = isLayerGroup ? expanded.has(path) : true;
  const className = node.attributes?.class;

  const classes = [
    "module-card",
    `depth-${Math.min(depth, 4)}`,
    `type-${typeClass(node.type)}`,
    isSelected ? "selected" : "",
    isMatch ? "match" : "",
    isDimmed ? "dimmed" : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <article className={classes}>
      <header
        style={{ cursor: "pointer" }}
        onClick={() => onSelectNode && onSelectNode(path)}
      >
        <div>
          <h3>{node.name}</h3>
          <span>{className ? `${node.type} · ${className}` : node.type}</span>
        </div>
        <div className="card-actions">
          {node.repeat && <b>x {node.repeat}</b>}
          {isLayerGroup && (
            <button
              className="layer-group-toggle card"
              onClick={(e) => {
                e.stopPropagation();
                onToggleGroup && onToggleGroup(path);
              }}
            >
              {isExpanded ? "Collapse" : "Expand"}
            </button>
          )}
        </div>
      </header>
      <AttributeGrid attributes={node.attributes} sourceFields={node.source_fields} />
      {node.children?.length > 0 && (!isLayerGroup || isExpanded) && (
        <div className="module-children">
          {node.children.map((child, index) => (
            <ModuleCards
              key={`${node.id}-${child.id}-${child.name}-${index}`}
              node={child}
              depth={depth + 1}
              path={pathChild(path, index)}
              selectedPath={selectedPath}
              matchedPaths={matchedPaths}
              expandedGroups={expandedGroups}
              searchActive={searchActive}
              onSelectNode={onSelectNode}
              onToggleGroup={onToggleGroup}
            />
          ))}
        </div>
      )}
    </article>
  );
}

export default ModuleCards;
