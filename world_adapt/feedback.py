"""Evaluator-only matching of predictions to actions the agent later executes."""
from dataclasses import dataclass


@dataclass
class _Query:
    query_id: str
    start: int
    actions: tuple
    observations: tuple[str, ...]
    compared: int = 0
    mismatches: int = 0
    closed: bool = False


class PredictionFeedback:
    def __init__(self):
        self._queries: list[_Query] = []
        self.first_mismatch_real_action: int | None = None

    def predict(self, query_id, start, actions, observations):
        self._queries.append(_Query(query_id, start, tuple(actions), tuple(observations)))

    def execute(self, game, trajectory):
        events = []
        for query in self._queries:
            while not query.closed and query.compared < len(query.observations):
                index = query.start + query.compared
                if index >= len(trajectory):
                    break
                record = trajectory[index]
                if record.action != query.actions[query.compared]:
                    query.closed = True
                    break
                mismatch = game.observe(record.next_state) != query.observations[query.compared]
                query.compared += 1
                query.mismatches += int(mismatch)
                if mismatch:
                    self.first_mismatch_real_action = min(index + 1, self.first_mismatch_real_action or index + 1)
                events.append({'query_id': query.query_id, 'real_action': index + 1,
                               'predicted_step': query.compared, 'visible_mismatch': mismatch})
        return events

    def summary(self):
        return {'matched_prediction_queries': sum(q.compared > 0 for q in self._queries),
                'matched_prediction_steps': sum(q.compared for q in self._queries),
                'observed_mismatch_queries': sum(q.mismatches > 0 for q in self._queries),
                'observed_mismatch_steps': sum(q.mismatches for q in self._queries),
                'first_mismatch_real_action': self.first_mismatch_real_action}
