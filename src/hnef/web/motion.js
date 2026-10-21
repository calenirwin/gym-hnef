/* Visual transitions never alter engine state. */
'use strict';
window.HnefMotion = (() => {
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const pending = new Set();
  let generation = 0;

  function stop() {
    generation += 1;
    for (const cleanup of [...pending]) cleanup();
  }

  function capture(board) {
    const pieces = new Map();
    for (const cell of board.querySelectorAll('.square')) {
      const piece = cell.querySelector('.piece');
      if (piece) pieces.set(`${cell.dataset.row},${cell.dataset.col}`, {
        node: piece.cloneNode(true), rect: piece.getBoundingClientRect(),
      });
    }
    return pieces;
  }

  function ghost(snapshot, destination, captured = false) {
    if (!snapshot) return Promise.resolve();
    const node = snapshot.node;
    node.className = 'motion-piece';
    node.setAttribute('aria-hidden', 'true');
    const rect = snapshot.rect;
    Object.assign(node.style, {
      position: 'fixed', left: `${rect.left}px`, top: `${rect.top}px`,
      width: `${rect.width}px`, height: `${rect.height}px`,
    });
    document.body.append(node);
    const target = destination?.getBoundingClientRect();
    if (destination) destination.style.visibility = 'hidden';
    const keyframes = captured
      ? [{opacity: 1, transform: 'scale(1)'}, {opacity: 0, transform: 'scale(.78) translateY(-8px)'}]
      : [{transform: 'translate(0,0)'}, {
        transform: `translate(${target.left - rect.left}px,${target.top - rect.top}px)`,
      }];
    const animation = node.animate(keyframes, {
      duration: captured ? 270 : 300,
      delay: captured ? 180 : 0,
      easing: 'cubic-bezier(.22,.8,.25,1)', fill: 'both',
    });
    return new Promise(resolve => {
      const cleanup = () => {
        if (!pending.delete(cleanup)) return;
        animation.cancel(); node.remove();
        if (destination) destination.style.visibility = '';
        resolve();
      };
      pending.add(cleanup);
      animation.onfinish = cleanup;
      animation.oncancel = cleanup;
    });
  }

  async function play(board, before, move) {
    if (reduced.matches || !before || !move) return;
    const current = generation;
    const destination = board.querySelector(
      `[data-row="${move.destination[0]}"][data-col="${move.destination[1]}"] .piece`,
    );
    if (!destination) return;
    await Promise.all([
      ghost(before.get(move.source.join(',')), destination),
      ...move.captured.map(square => ghost(before.get(square.join(',')), null, true)),
    ]);
    return current === generation;
  }

  addEventListener('resize', stop);
  addEventListener('scroll', stop, {passive: true});
  reduced.addEventListener('change', stop);
  return {capture, play, stop};
})();
