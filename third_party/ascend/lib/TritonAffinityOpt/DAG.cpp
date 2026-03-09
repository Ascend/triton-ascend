#include "TritonAffinityOpt/DAG.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/CopyOpInterface.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

namespace mlir { namespace AffinityDAG {

const auto printFlags = OpPrintingFlags()
                                    .enableDebugInfo(true, true)
                                    .skipRegions();

const char* literalCoreType(CoreType ct) {
  switch (ct) {
    case VECTOR:
      return "VECTOR";
    case CUBE:
      return "CUBE";
    case SCALAR:
      return "SCALAR";
    case UNDETERMINED:
      return "UNDETERMINED";
  }
  return "Unknown";
}

const char* Graph::getCoreTypeStrOf(Value value) {
  if (valueTypes->contains(value)) {
    return literalCoreType((*valueTypes)[value]);
  }
  return "Unknown";
}

Node::Node(
  Operation* const op,
  Graph* const graph
)
:
  op(op),
  graph(graph),
  nodeType(NodeType::NT_Default)
{
  auto block = graph->block;

  if (!op) {
    return;
  }

  for (auto operand : op->getOperands()) {
    auto definingOp = operand.getDefiningOp();
    if (!definingOp) {
      continue;
    }
    auto definingNode = graph->opMap->lookup(definingOp);
    if (!definingNode) {
      continue;
    }
    this->ins.insert(definingNode);
  }
}

// std::pair<Node*, Node*> Node::SyncNode(
//   Operation* const send,
//   Operation* const receive,
//   Graph* const sendGraph,
//   Graph* const receiveGraph
// ) {
//   auto hivmOp = llvm::dyn_cast<CopyOpInterface>(send);

//   if (!hivmOp) {
//     assert(hivmOp);
//     llvm::errs() << "[Warning] The following operation does not implement llvm copy interface.\n";
//     send->dump();
//     return std::make_pair(new Node(send, sendGraph), new Node(receive, receiveGraph));
//   }

//   auto sendNode = new Node(send, sendGraph, NodeType::NT_Sync);
//   auto receiveNode = new Node(receive, receiveGraph);

//   if (auto sourceOp = hivmOp.getSource().getDefiningOp()) {
//     if (auto sourceNode = sendGraph->opMap->lookup(sourceOp)) {
//       sendNode->ins.insert(sourceNode);
//     }
//   }

//   sendNode->outs.insert(receiveNode);
//   receiveNode->ins.insert(sendNode);

//   return std::make_pair(sendNode, receiveNode);
// }

bool opIsScf(Operation* op) {
  return llvm::isa<scf::SCFDialect>(op->getDialect());
}

Graph::Graph(
  Block* block, Graph* parent,
  llvm::DenseMap<Operation*, Node*>* opMapDefault,
  llvm::DenseMap<Value, CoreType>* valueTypesDefault,
  bool noDependencies
  ):
  block(block), parentGraph(parent), opMap(opMapDefault), valueTypes(valueTypesDefault)
{
  auto blockOp = block->getParentOp();

  if (this->opMap == nullptr) {
    this->opMap = new llvm::DenseMap<Operation*, Node*>;
  }

  if (this->valueTypes == nullptr) {
    this->valueTypes = new llvm::DenseMap<Value, CoreType>;
  }

  // if (this->markedByCubeLoad == nullptr) {
  //   this->markedByCubeLoad = new llvm::DenseSet<Value>;
  // }

  // if (this->usedBy == nullptr) {
  //   this->usedBy = new llvm::DenseMap<Value, CoreType>;
  // }

  auto& opMap = *this->opMap;
  auto& valueTypes = *this->valueTypes;

  Node* terminator = nullptr;
  for(auto& op: *block) {
    Node* node = new Node(&op, this);
    opMap[&op] = node;
    if (opIsScf(&op)) {
      for(auto& region : op.getRegions()) {
        for(auto& subblock : region.getBlocks()) {
          auto subgraph = Graph::inheritConstructFrom(&subblock, this);
          node->subgraphs.push_back(subgraph);
        }
      }
    }

    if (noDependencies) {
      continue;
    }

    for(const auto& operand : op.getOperands()) {

      int resultNum = 0;
      auto definingOp = operand.getDefiningOp();

      // block args
      if (!definingOp) {
        continue;
      }

      if (!opMap.contains(definingOp)) {
        llvm::errs() << "The following op is not found in graph or any parent graphs! \n";
        definingOp->dump();
      }

      auto definingNode = opMap[definingOp];
      auto definingBlock = definingNode->graph->block;
      auto currOp = &op;

      while (currOp && currOp->getBlock() != definingBlock) {
        auto block = currOp->getBlock();
        if (!block) {
          break;
        }
        if (llvm::isa<triton::FuncOp>(block->getParentOp())) {
          break;
        }
        currOp = block->getParentOp();
      }

      if (!(currOp && opMap.contains(currOp))) {
        llvm::errs() << "The following op is not in opMap!\n";
        currOp->print(llvm::errs(), printFlags);
        llvm::errs() << "\n\n";
      }

      definingNode->outs.insert(opMap[currOp]);
    }
  }

  if (block->mightHaveTerminator()) {
    if (auto terminator = block->getTerminator()) {
      this->terminator = opMap[terminator];
    }
  }
}

bool valueIsScalar(Value value) {
  auto type = value.getType();

  if (type.isIntOrIndexOrFloat()) {
    return true;
  }

  if (auto tensorType = llvm::dyn_cast<RankedTensorType>(type)) {
    return tensorType.getRank() == 0;
  }

  if (auto _ = llvm::dyn_cast<triton::PointerType>(type)) {
    return true;
  }

  return false;
}

bool valueIsTensorOfPtr(Value value) {
  auto type = value.getType();
  if (auto tensorType = llvm::dyn_cast<RankedTensorType>(type)) {
    auto elementType = tensorType.getElementType();
    if (llvm::isa<triton::PointerType>(elementType)) {
      return true;
    }
  }

  return false;
}

/**
  * @brief Determines the core type of a certain op
  * Specifically, only dot is recognised as CUBE, load and trans are interpreted as SCALAR here
  * load and trans operations are marked as CUBE when necessariy afterwards
  */
CoreType coreTypeOf(Operation* op) {
  return llvm::TypeSwitch<Operation*, CoreType>(op)
    .Case<triton::DotOp>([](auto) {
      return CUBE;
    })
    .Case<arith::ConstantOp, triton::AdvanceOp, triton::TransOp>([](auto) {
      return SCALAR;
    })
    .Case<arith::SelectOp>([](arith::SelectOp op) {
      // when cond is vector, selectOp should be vector, otherwise scalar
      return (
        valueIsScalar(op.getConditionMutable().get()) ? SCALAR : VECTOR
      );
    })
    .Default([](Operation* op) {
      auto isVector = false;
      for(auto operand : op->getOperands()) {
        if (!valueIsScalar(operand)) {
          // if (valueIsTensorOfPtr(operand)) {
          //   return SCALAR;
          // }
          isVector = true;
        }
      }

      for(auto result : op->getResults()) {
        if (!valueIsScalar(result)) {
          // if (valueIsTensorOfPtr(result)) {
          //   return SCALAR;
          // }
          isVector = true;
        }
      }

      if (isVector) {
        return VECTOR;
      }

      return SCALAR;
    });
}

void Graph::markAsCoreType(Operation* op, CoreType ct) {
  for (Value result : op->getResults()) {
    (*valueTypes)[result] = ct;
  }
}

void Graph::markCubeLoadUpstream(Operation* op) {
  auto coreType = coreTypeOf(op);
  if (coreType != SCALAR && coreType != VECTOR) {
    llvm::dbgs() << "Marked as Cube in markCubeLoadUpstream: " << *op;
    markAsCoreType(op, CUBE);
  } else {
    llvm::dbgs() << "Marked as scalar in markCubeLoadUpstream: " << *op;
    markAsCoreType(op, SCALAR);
  }

  markedAsCube.insert(op);
  for(auto operand : op->getOperands()) {
    llvm::TypeSwitch<Value, void>(operand)
      .Case<OpResult>([&](auto) {
        auto defOp = operand.getDefiningOp();
        if (!markedAsCube.contains(operand.getDefiningOp()) && !opIsScf(defOp))
          markCubeLoadUpstream(operand.getDefiningOp());
      })
      .Case<BlockArgument>([&](BlockArgument blockArg) {
        if (valueIsScalar(blockArg)) {
          return;
        }
        auto block = blockArg.getOwner();
        if (!terminator || !block->getParentOp()) {
          return;
        }

        auto loopOp = llvm::dyn_cast<LoopLikeOpInterface>(this->block->getParentOp());
        if (!loopOp) {
          return;
        }
        auto yieldOp = terminator->op;
        auto yieldOperand = loopOp.getTiedLoopYieldedValue(blockArg);
        if (!yieldOperand) {
          return;
        }
        auto yieldVal = yieldOperand -> get();

        (*valueTypes)[yieldVal] = CUBE;

        if (auto op = yieldVal.getDefiningOp()) {
          markCubeLoadUpstream(op);
        }
      });
  }
}

void Graph::markDotUpstream(Operation* op) {
  markAsCoreType(op, CUBE);
  llvm::dbgs() << "Marked as Cube in markDotUpstream: " << *op;
  markedAsCube.insert(op);
  for(auto operand : op->getOperands()) {
    auto defOp = operand.getDefiningOp();
    if (defOp) {
      if (!markedAsCube.contains(defOp)) {
        llvm::TypeSwitch<Operation*, void>(defOp)
          .Case<triton::TransOp>([&](auto) {
            markDotUpstream(defOp);
          })
          .Case<triton::LoadOp>([&](auto) {
            markCubeLoadUpstream(defOp);
          });
      }
    } else if (!valueIsScalar(operand)) {
      (*valueTypes)[operand] = CUBE;
    }
  }
}

void Graph::markCore() {
  auto& valueTypes = *this->valueTypes;
  llvm::DenseSet<Node*> marked;

  for(auto& opRef : *this->block) {
    auto op = &opRef;
    auto node = (*this->opMap)[op];


    if (llvm::isa<triton::DotOp>(op)) {
      markDotUpstream(op);
      continue;
    } else if (llvm::isa<triton::StoreOp>(op) && valueTypes[op->getOperand(1)] == CUBE) {
      markCubeLoadUpstream(op);
      continue;
    }

    if (opIsScf(op)) {
      for(auto subgraph : node->subgraphs) {
        subgraph->markCore();
        if(!subgraph->terminator) {
          continue;
        }
        if(auto yieldOp = llvm::dyn_cast_if_present<scf::YieldOp>(subgraph->terminator->op)) {
          for(auto [result, yieldVal] : llvm::zip_equal(op->getResults(), yieldOp->getOperands())) {
            valueTypes[result] = valueTypes[yieldVal];
          }
        }
      }
      continue;
    }

    auto coreType = coreTypeOf(op);
    markAsCoreType(op, coreType);
  }

  if (auto forOp = llvm::dyn_cast_if_present<scf::ForOp>(this->block->getParentOp())) {
    if (!this->terminator) {
      return;
    }
    if (auto yieldOp = llvm::dyn_cast_if_present<scf::YieldOp>(this->terminator->op)) {
      for(auto [iterArg, yieldVal] : llvm::zip_equal(forOp.getRegionIterArgs(), yieldOp.getOperands())) {
        valueTypes[iterArg] = valueTypes[yieldVal];
      }
    }
  }

  if (auto whileOp = llvm::dyn_cast_if_present<scf::WhileOp>(this->block->getParentOp())) {
    llvm_unreachable("while op is not supported!");
  }
}

bool valueDefinedInBlock(Block* block, Value value) {
  auto blockOp = block->getParentOp();
  return llvm::TypeSwitch<Value, bool>(value)
                .Case<BlockArgument>([&](BlockArgument blockArg) {
                  auto valueBlock = blockArg.getParentBlock();
                  return (block == valueBlock)
                          || (block->findAncestorOpInBlock(*valueBlock->getParentOp()) != nullptr);
                })
                .Case<OpResult>([&](OpResult result) {
                  auto definingOp = value.getDefiningOp();
                  return blockOp->isAncestor(definingOp);
                })
                .Default(false);
}

void Node::convertToSubGraph() {
  Block* subBlock = new Block();
  subBlock->getOperations().push_back(op);

  auto& opMap = *this->graph->opMap;
  auto& valueTypes = *this->graph->valueTypes;

  Graph* subgraph = new Graph(subBlock, graph, &opMap, &valueTypes, true);
  this->subgraphs.push_back(subgraph);
}

void Node::flattenSubGraph(size_t index) {
  if (index >= subgraphs.size()) return;

  Graph* subGraph = subgraphs[index];
  Block* subBlock = subGraph->block;
  Graph* targetGraph = this->graph;
  Block* targetBlock = targetGraph->block;

  Block::iterator targetIt = this->op->getIterator();

  while (&subBlock->front() != this->op) {
    Operation* opToMove = &subBlock->front();
    opToMove->moveBefore(targetBlock, targetIt);
  }

  auto subGraphIt = std::next(subBlock->begin());
  targetIt = std::next(targetIt);

  while (subGraphIt != subBlock->end()) {
    Operation* opToMove = &*subGraphIt;
    subGraphIt = std::next(subGraphIt);

    opToMove->moveBefore(targetBlock, targetIt);
  }

  subgraphs.erase(subgraphs.begin() + index);
  delete subBlock;
  delete subGraph;
}

void Graph::updateDependencies(Node* node) {
  if (!node) return;

  node->ins.clear();

  auto findNode = [&](Operation* targetOp) -> Node* {
    Graph* current = this;
    while (current) {
      if (current->opMap) {
        auto it = current->opMap->find(targetOp);
        if (it != current->opMap->end()) return it->second;
      }
      current = current->parentGraph;
    }
    return nullptr;
  };

  for (Value operand : node->op->getOperands()) {
    if (Node* defNode = findNode(operand.getDefiningOp())) {
      node->ins.insert(defNode);
      defNode->outs.insert(node);
    }
  }

  for (Value result : node->op->getResults()) {
    for (Operation* user : result.getUsers()) {
      if (Node* userNode = findNode(user)) {
        node->outs.insert(userNode);
        userNode->ins.insert(node);
      }
    }
  }
}

Node* Graph::insertOp(Operation* op, std::optional<Operation*> positionPos) {
  if (!op || !this->block) return nullptr;

  Block::iterator insertPt;
  if (positionPos && positionPos.value()->getBlock() == this->block) {
    insertPt = positionPos.value()->getIterator();
  } else {
    Operation* term = this->block->getTerminator();
    insertPt = term ? term->getIterator() : this->block->end();
  }

  op->moveBefore(this->block, insertPt);

  Node*& nodePtr = (*this->opMap)[op];
  if (!nodePtr) {
    nodePtr = new Node(op, this);
  }

  updateDependencies(nodePtr);
  return nodePtr;
}

std::tuple<
  std::shared_ptr<Graph>,
  // std::unique_ptr<Node>,
  // std::unique_ptr<Graph::OpMapRaw>,
  // std::unique_ptr<Graph::ValueTypesRaw>
  Node*
> Graph::fromMultiBlockFunc(triton::FuncOp funcOp) {
  // auto opMap = std::make_unique<OpMapRaw>();
  // auto valueTypes = std::make_unique<ValueTypesRaw>();
  // auto dummyBlock = new Block();
  // auto dummyGraph = std::make_shared<Graph>(dummyBlock, nullptr, opMap.get(), valueTypes.get());
  // auto dummyNode = std::make_unique<Node>(nullptr, dummyGraph.get());

  auto dummyBlock = new Block();
  auto dummyGraph = std::make_shared<Graph>(dummyBlock);
  auto dummyNode = new Node(nullptr, dummyGraph.get());
  size_t maxIterCount = 0;

  for (auto& block : funcOp.getBody()) {
    auto subgraph = new Graph(
      &block,
      dummyGraph.get(),
      dummyGraph->opMap,
      dummyGraph->valueTypes
    );
    dummyNode->subgraphs.push_back(subgraph);
    subgraph -> markCore();
  }

  // return {
  //   std::move(dummyGraph),
  //   std::move(dummyNode),
  //   std::move(opMap),
  //   std::move(valueTypes)
  // };
  //

  return {
    dummyGraph,
    dummyNode
  };
};

} }