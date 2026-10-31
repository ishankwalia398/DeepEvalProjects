-- Analysis queries over reports/benchmark.db. Each block starts with "-- name:".
-- Named parameters (:run_a, :run_b) are bound by src/report.py; set them yourself in a SQL client.

-- name: error_rate_by_severity
-- 1. Severity error rate by gold severity and prompt variant (triage runs only).
SELECT r.prompt_variant,
       i.gold_severity,
       COUNT(*) AS n,
       ROUND(1.0 - AVG(s.value), 3) AS error_rate
FROM scores s
JOIN runs r  ON r.run_id = s.run_id
JOIN items i ON i.item_id = s.item_id
WHERE s.metric = 'severity_correct'
GROUP BY r.prompt_variant, i.gold_severity
ORDER BY r.prompt_variant, i.gold_severity;

-- name: regressed_items
-- 2. Labels that were correct in run A and wrong in run B.
SELECT a.item_id,
       REPLACE(a.metric, '_correct', '') AS field,
       pa.severity || '/' || pa.component || '/' || pa.root_cause AS prediction_a,
       COALESCE(pb.severity || '/' || pb.component || '/' || pb.root_cause, 'parse error') AS prediction_b
FROM scores a
JOIN scores b       ON b.item_id = a.item_id AND b.metric = a.metric AND b.run_id = :run_b
JOIN predictions pa ON pa.run_id = a.run_id AND pa.item_id = a.item_id
JOIN predictions pb ON pb.run_id = b.run_id AND pb.item_id = b.item_id
WHERE a.run_id = :run_a
  AND a.metric IN ('severity_correct', 'component_correct', 'root_cause_correct')
  AND a.value = 1 AND b.value = 0
ORDER BY a.item_id, field;

-- name: hallucination_by_platform
-- 3. Hallucinated-citation rate by platform and prompt variant.
SELECT r.prompt_variant,
       i.platform,
       CAST(SUM(CASE WHEN s.metric = 'citations_invalid' THEN s.value ELSE 0 END) AS INTEGER) AS invalid,
       CAST(SUM(CASE WHEN s.metric = 'citations_total' THEN s.value ELSE 0 END) AS INTEGER) AS total,
       ROUND(SUM(CASE WHEN s.metric = 'citations_invalid' THEN s.value ELSE 0 END)
             / NULLIF(SUM(CASE WHEN s.metric = 'citations_total' THEN s.value ELSE 0 END), 0), 3) AS hallucinated_rate
FROM scores s
JOIN runs r  ON r.run_id = s.run_id
JOIN items i ON i.item_id = s.item_id
WHERE s.metric IN ('citations_invalid', 'citations_total')
GROUP BY r.prompt_variant, i.platform
ORDER BY r.prompt_variant, i.platform;
