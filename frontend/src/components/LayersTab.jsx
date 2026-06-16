import EmptyState from "./EmptyState";
import ModuleCards from "./ModuleCards";

function LayersTab({
  structure,
  selectedPath,
  matchedPaths,
  expandedGroups,
  searchActive,
  onSelectNode,
  onToggleGroup,
  onExpandAllGroups,
  onCollapseAllGroups,
}) {
  return (
    <section className="layers-panel">
      <div className="panel-toolbar compact">
        <h2>Layers</h2>
        <div className="toolbar-actions">
          <button onClick={onExpandAllGroups} disabled={!structure}>
            Expand all
          </button>
          <button onClick={onCollapseAllGroups} disabled={!structure}>
            Collapse all
          </button>
        </div>
      </div>
      {structure ? (
        <ModuleCards
          node={structure.root}
          depth={0}
          path="root"
          selectedPath={selectedPath}
          matchedPaths={matchedPaths}
          expandedGroups={expandedGroups}
          searchActive={searchActive}
          onSelectNode={onSelectNode}
          onToggleGroup={onToggleGroup}
        />
      ) : (
        <EmptyState />
      )}
    </section>
  );
}

export default LayersTab;
