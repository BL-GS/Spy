#pragma once

#include <cstddef>

namespace spy {
	
	enum class ModelType: size_t { 
		LLaMa, Falcon, GPT2,
		ModelTypeEnd
	};

	enum class ModelRopeScalingType : int {
		Unspeicific = -1,
		None        = 0,
		Linear      = 1,
		Yarn        = 2
	};

	enum class ModelPoolingType {
		Unspecific = -1,
		None       = 0,
		Mean       = 1,
		Cls        = 2
	};

	enum class GrammerType: int {
		/// End of relu defintion
		EndOfRuleDefinition        = 0,
		/// Start of alternate definition for rule
		AlternateDefinitionForRule = 1,
		/// Non-terminal element: reference to rule
		ReferenceToRelu            = 2,
		/// Terminal element: character (code point)
		Character                  = 3,
		/// Inverse character  ([^a], [^a-b], [^abc])
		InverseCharacter           = 4,
		/// Modifies a preceding `Character` or `InverseCharacter` to be an inclusive range ([a-z])
		RangedCharacter            = 5,
		/// modifies a preceding `Character` or `RangedCharacter` to add an alternate char to match ([ab], [a-zA])
		AlternateCharacter         = 6
	};

	enum class ModelTensorType: size_t {
		Tokenembedding = 0, TokenembeddingNorm, TokenTypes, Positionembedding,

		Output, OutputNorm, RopeFrequency, LayerOutNorm,

		AttentionQ, AttentionK, AttentionV, AttentionQKV, 
		AttentionQNorm, AttentionKNorm,
		AttentionOutput, AttentionNorm, AttentionNorm2,
		AtentionOutNorm, AttentionRotationembedding,

		FFNGateInp, FFNNorm, FFNDown, FFNGate, FFNUp, FFNActivation, 
		FFNDownExp, FFNGateExp, FFNUpExp,

		ModelTensorTypeEnd
	};
	
}  // namespace spy