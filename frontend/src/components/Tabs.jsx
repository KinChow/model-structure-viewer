function Tabs({ items, active, onChange }) {
  return (
    <div className="tabs" role="tablist">
      {items.map((item) => (
        <button
          key={item}
          className={active === item ? "active" : ""}
          onClick={() => onChange(item)}
          role="tab"
        >
          {item}
        </button>
      ))}
    </div>
  );
}

export default Tabs;
