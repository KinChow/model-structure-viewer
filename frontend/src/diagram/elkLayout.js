/**
 * Historical compatibility entry point.
 *
 * The viewer now has one production layout pipeline: Layout IR projected into
 * a single ELK compound graph with ELK orthogonal edge routing. Keeping this
 * alias avoids breaking older imports without retaining a second router.
 */
export { layoutGraphWithElkOnly as layoutGraphWithElk } from "./elkOnlyLayout.js";
