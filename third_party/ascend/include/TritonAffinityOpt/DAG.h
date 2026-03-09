#ifndef AffinityDAGDEF
#define AffinityDAGDEF
#include "Utils.hpp"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TinyPtrVector.h"
#include <cassert>
#include <cstddef>
#include <memory>

namespace mlir { namespace AffinityDAG {

/**
  * @brief Mark the type of device of a certain Op or result
  *
  * This enum is reused for both values and operations
*/
enum CoreType {
  VECTOR = 1 << 0,
  CUBE = 1 << 1,
  SCALAR = VECTOR | CUBE,
  UNDETERMINED = 0,
  // UNDETERMINED = 0,
  // VECTOR = 1 << 0,
  // CUBE = 1 << 1,
  // SCALAR = VECTOR | CUBE,
};

constexpr inline CoreType operator| (CoreType lhs, CoreType rhs) {
  return enumOp(std::bit_or<>(), lhs, rhs);
}

inline CoreType operator& (CoreType lhs, CoreType rhs) {
  return enumOp(std::bit_and<>(), lhs, rhs);
}

inline bool intersects(CoreType lhs, CoreType rhs) {
  return (lhs & rhs) != UNDETERMINED;
}

const char* literalCoreType(CoreType ct);

class Node;

class Graph {
  friend class Node;

public:
  using OpMapRaw = llvm::DenseMap<Operation*, Node*>;
  using ValueTypesRaw = llvm::DenseMap<Value, CoreType>;
  using OpMap = OpMapRaw*;
  using ValueTypes = ValueTypesRaw*;
  Block* block;
  OpMap opMap;
  ValueTypes valueTypes;
  Node* terminator = nullptr;
  Graph* parentGraph = nullptr;
  llvm::DenseSet<Operation*> markedAsCube;

  explicit Graph(Block* blk, Graph* parentGraph = nullptr,
        OpMap defaultOpMap = nullptr,
        ValueTypes defaultValueTypes = nullptr,
        bool noDependencies = false);

  static Graph* inheritConstructFrom(Block* blk, Graph* parentGraph) {
    return new Graph(
      blk, parentGraph,
      parentGraph->opMap,
      parentGraph->valueTypes
    );
  };

  ~Graph() {}

  Node* insertOp(Operation* op, std::optional<Operation*> positionPos = std::nullopt);

  void markCore();
  const char* getCoreTypeStrOf(Value value);

  /**
   * @brief Initialise a dummy graph with a single dummy Node that contains all blocks of the function as subgraphs
   */
  static std::tuple<
    std::shared_ptr<Graph>,
    // std::unique_ptr<Node>,
    // std::unique_ptr<OpMapRaw>,
    // std::unique_ptr<ValueTypesRaw>
    Node*
  > fromMultiBlockFunc(triton::FuncOp func);

private:
  void updateDependencies(Node* node);
  void markAsCoreType(Operation* op, CoreType ct);

  void markDotUpstream(Operation* op);
  void markCubeLoadUpstream(Operation* op);

  // CoreType diffuseFromUpstream(OpResult result);
  // bool diffuseAcross(Operation* op);
  // bool updateValueType(Value value, CoreType newCoreType);
  // CoreType diffusedYieldValueFromIterArgs(scf::YieldOp yieldOp);

  // /**
  //  * @brief Try to diffuse the core types of current graph and the subgraphs, single pass
  //  * @return Whether this iteration has changed any valueTypes
  //  */
  // bool markCoreSingleStep();
};

class Node {
  friend class Graph;
public:
  enum NodeType {
    NT_Default,
    NT_Sync,
  };
  const NodeType nodeType;

  Operation* op;
  Graph* graph; // 所属图
  llvm::SmallPtrSet<Node*, 4> ins;
  llvm::SmallPtrSet<Node*, 2> outs;

  llvm::TinyPtrVector<Graph*> subgraphs;

  Graph* getSubGraph(size_t index = 0) {
    return index >= subgraphs.size() ? nullptr : subgraphs[index];
  }

  bool hasSubGraph() const { return !subgraphs.empty(); }
  void convertToSubGraph();
  void flattenSubGraph(size_t index = 0);

  /**
    * @brief Initialize a default node directly under the graph, together with its ins.
    */
  Node(Operation* const op, Graph* const graph);

  /**
   * @brief Initialize a pair of sync node directly under the graph, handling syncronising scross cores.
   */
  // static std::pair<Node*, Node*> SyncNode(
  //   Operation* const send,
  //   Operation* const receive,
  //   Graph* const sendGraph,
  //   Graph* const receiveGraph
  // );

private:
  /**
  * @brief Initialize a node with the specified node type, without handling dependencies
  */
  Node(Operation* const op, Graph* const graph, NodeType nt) : op(op), graph(graph), nodeType(nt) {};
};


class GraphManager {
private:
  llvm::DenseMap<llvm::StringRef, std::shared_ptr<AffinityDAG::Graph>> graphs;

public:
  static GraphManager &getInstance() {
    static GraphManager instance;
    return instance;
  }

  void registerGraph(llvm::StringRef funcName, std::shared_ptr<AffinityDAG::Graph> graph) {
    graphs[funcName] = graph;
  }

  AffinityDAG::Graph* getGraph(llvm::StringRef funcName) {
    auto it = graphs.find(funcName);
    return it != graphs.end() ? it->second.get() : nullptr;
  }

  void removeGraph(llvm::StringRef funcName) {
    graphs.erase(funcName);
  }
};

} }
#endif