#include <cassert>
#include <iostream>
#include <string>

#include "controller/EvidenceFeatures.hpp"
#include "controller/GraphSelector.hpp"

namespace {

mobile_rag::GraphSelector::Availability make_availability(bool sqlite_available,
                                                          bool lexical_available,
                                                          bool semantic_available) {
  mobile_rag::GraphSelector::Availability availability;
  availability.sqlite_available = sqlite_available;
  availability.lexical_graph_available = lexical_available;
  availability.semantic_hash_graph_available = semantic_available;
  return availability;
}

void test_prefers_lexical_for_term_rich_queries() {
  mobile_rag::GraphSelector selector({true, 3, 0.15f, 0.50f});

  const auto decision = selector.choose_initial_graph(
      "sqlite metadata retrieval traces",
      make_availability(true, true, true));

  assert(decision.graph == mobile_rag::RetrievalGraph::LEXICAL_PREFILTER);
  assert(decision.reason == "term_rich_query");
  assert(!decision.escalated);
}

void test_prefers_lexical_hash_for_numeric_queries() {
  mobile_rag::GraphSelector selector({true, 3, 0.15f, 0.50f});

  const auto decision = selector.choose_initial_graph(
      "sqlite metadata 2024 traces",
      make_availability(true, true, true));

  assert(decision.graph == mobile_rag::RetrievalGraph::LEXICAL_HASH_PREFILTER);
  assert(decision.reason == "numeric_query");
  assert(!decision.escalated);
}

void test_escalates_to_lexical_hash_when_evidence_is_weak() {
  mobile_rag::GraphSelector selector({true, 3, 0.15f, 0.50f});

  mobile_rag::EvidenceFeatures evidence;
  evidence.score_margin = 0.04f;
  evidence.coverage_ratio = 0.25f;
  evidence.retrieved_chunk_count = 1;

  const auto decision = selector.maybe_escalate(
      mobile_rag::RetrievalGraph::SEMANTIC_HASH_PREFILTER,
      make_availability(true, true, true),
      evidence);

  assert(decision.graph == mobile_rag::RetrievalGraph::LEXICAL_HASH_PREFILTER);
  assert(decision.reason == "low_evidence_upgrade");
  assert(decision.escalated);
}

void test_keeps_graph_when_evidence_is_sufficient() {
  mobile_rag::GraphSelector selector({true, 3, 0.15f, 0.50f});

  mobile_rag::EvidenceFeatures evidence;
  evidence.score_margin = 0.42f;
  evidence.coverage_ratio = 1.0f;
  evidence.retrieved_chunk_count = 2;

  const auto decision = selector.maybe_escalate(
      mobile_rag::RetrievalGraph::LEXICAL_PREFILTER,
      make_availability(true, true, true),
      evidence);

  assert(decision.graph == mobile_rag::RetrievalGraph::LEXICAL_PREFILTER);
  assert(decision.reason == "evidence_sufficient");
  assert(!decision.escalated);
}

}  // namespace

int main() {
  test_prefers_lexical_for_term_rich_queries();
  test_prefers_lexical_hash_for_numeric_queries();
  test_escalates_to_lexical_hash_when_evidence_is_weak();
  test_keeps_graph_when_evidence_is_sufficient();
  std::cout << "GraphSelector tests passed\n";
  return 0;
}
