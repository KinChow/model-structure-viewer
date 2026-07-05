function ShapeFlow({ attributes }) {
  const inputShape = attributes?.input_shape;
  const outputShape = attributes?.output_shape;
  if (!inputShape && !outputShape) return null;

  return (
    <div className="shape-flow">
      {inputShape && (
        <span>
          <em>input</em>
          <strong>{inputShape}</strong>
        </span>
      )}
      {inputShape && outputShape && <b aria-hidden="true">→</b>}
      {outputShape && (
        <span>
          <em>output</em>
          <strong>{outputShape}</strong>
        </span>
      )}
    </div>
  );
}

export default ShapeFlow;
