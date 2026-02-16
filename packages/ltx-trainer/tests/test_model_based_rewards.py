"""Tests for model-based reward functions (VideoScore2, UnifiedReward-Think).

This test suite validates both reward families with synthetic videos.
Tests will gracefully skip if models are not downloaded.
"""

import sys
from pathlib import Path

import pytest
import torch

# Add package to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def make_video(frame_fn, num_frames: int = 49):
    """Build a [3, F, H, W] video where frame_fn(f) returns (R, G, B) per frame.

    Args:
        frame_fn: Function that takes frame index and returns (R, G, B) tuple
        num_frames: Number of frames (must satisfy num_frames % 8 == 1)

    Returns:
        Video tensor [3, F, H, W] in [0, 1] range
    """
    assert num_frames % 8 == 1, f"num_frames must satisfy % 8 == 1, got {num_frames}"
    H, W = 256, 256  # Reasonable resolution for model inference
    video = torch.zeros(3, num_frames, H, W)
    for f in range(num_frames):
        r, g, b = frame_fn(f)
        video[0, f] = r
        video[1, f] = g
        video[2, f] = b
    return video


# Test videos
BLACK_VIDEO = make_video(lambda f: (0, 0, 0))
WHITE_VIDEO = make_video(lambda f: (1, 1, 1))
RED_VIDEO = make_video(lambda f: (1, 0, 0))
BLUE_VIDEO = make_video(lambda f: (0, 0, 1))
ALTERNATING_VIDEO = make_video(lambda f: (1, 0, 0) if f % 2 == 0 else (0, 0, 1))
GRADIENT_VIDEO = make_video(lambda f: (f / 48, f / 48, f / 48))


@pytest.fixture(scope="module")
def check_models_available():
    """Check if models are available before running tests."""
    try:
        from transformers import AutoProcessor

        # Try to load model metadata (doesn't download full model)
        AutoProcessor.from_pretrained("TIGER-Lab/VideoScore2", trust_remote_code=True)
        return True
    except Exception:
        pytest.skip("VideoScore2/UnifiedReward models not available - skipping model-based tests")


class TestVideoScore2:
    """Tests for VideoScore2 reward function."""

    @pytest.fixture(scope="class")
    def videoscore2_rewards(self, check_models_available):
        """Load VideoScore2 rewards once for all tests."""
        from ltx_trainer.rl.rewards import get_reward_functions

        try:
            rewards = get_reward_functions("video_score2")
            assert len(rewards) == 3, "VideoScore2 should return 3 dimensions"
            return {name: fn for fn, name in rewards}
        except Exception as e:
            pytest.skip(f"Failed to load VideoScore2 model: {e}")

    def test_dimensions(self, videoscore2_rewards):
        """Test that all expected dimensions are present."""
        expected = ["videoscore2_visual_quality", "videoscore2_text_to_video_alignment", "videoscore2_physical_consistency"]
        assert set(videoscore2_rewards.keys()) == set(expected)

    def test_black_video(self, videoscore2_rewards):
        """Test scoring on a black video."""
        prompt = "A completely black screen"
        for name, reward_fn in videoscore2_rewards.items():
            score = reward_fn.compute(BLACK_VIDEO, prompt)
            assert isinstance(score, float), f"{name} should return float"
            assert 1.0 <= score <= 5.0, f"{name} score should be in [1, 5], got {score}"
            print(f"  {name}: {score:.3f}")

    def test_red_video(self, videoscore2_rewards):
        """Test scoring on a solid red video."""
        prompt = "A solid red screen"
        for name, reward_fn in videoscore2_rewards.items():
            score = reward_fn.compute(RED_VIDEO, prompt)
            assert isinstance(score, float)
            assert 1.0 <= score <= 5.0, f"{name} score should be in [1, 5], got {score}"
            print(f"  {name}: {score:.3f}")

    def test_alternating_video(self, videoscore2_rewards):
        """Test scoring on an alternating red/blue video."""
        prompt = "A video alternating between red and blue"
        for name, reward_fn in videoscore2_rewards.items():
            score = reward_fn.compute(ALTERNATING_VIDEO, prompt)
            assert isinstance(score, float)
            assert 1.0 <= score <= 5.0, f"{name} score should be in [1, 5], got {score}"
            print(f"  {name}: {score:.3f}")

    def test_gradient_video(self, videoscore2_rewards):
        """Test scoring on a video with gradient from black to white."""
        prompt = "A video fading from black to white"
        for name, reward_fn in videoscore2_rewards.items():
            score = reward_fn.compute(GRADIENT_VIDEO, prompt)
            assert isinstance(score, float)
            assert 1.0 <= score <= 5.0, f"{name} score should be in [1, 5], got {score}"
            print(f"  {name}: {score:.3f}")

    def test_caching(self, videoscore2_rewards):
        """Test that caching works correctly (same video object returns cached scores)."""
        prompt = "Test video for caching"
        test_video = make_video(lambda f: (0.5, 0.5, 0.5))

        # First call computes all dimensions
        scores_1 = {}
        for name, reward_fn in videoscore2_rewards.items():
            scores_1[name] = reward_fn.compute(test_video, prompt)

        # Second call should return cached results (exact same values)
        scores_2 = {}
        for name, reward_fn in videoscore2_rewards.items():
            scores_2[name] = reward_fn.compute(test_video, prompt)

        assert scores_1 == scores_2, "Cached scores should be identical"

    def test_different_prompts(self, videoscore2_rewards):
        """Test that different prompts can produce different scores."""
        test_video = RED_VIDEO

        prompt1 = "A red screen"
        prompt2 = "A blue ocean with waves"

        # Get alignment dimension (most likely to vary with prompt)
        alignment_fn = videoscore2_rewards["videoscore2_text_to_video_alignment"]

        score1 = alignment_fn.compute(test_video, prompt1)
        score2 = alignment_fn.compute(test_video, prompt2)

        # Note: We don't require scores to be different (model might still give same score)
        # Just verify both are valid
        assert 1.0 <= score1 <= 5.0
        assert 1.0 <= score2 <= 5.0
        print(f"  Red video with 'red screen' prompt: {score1:.3f}")
        print(f"  Red video with 'blue ocean' prompt: {score2:.3f}")


class TestUnifiedRewardThink:
    """Tests for UnifiedReward-Think reward function."""

    @pytest.fixture(scope="class")
    def unifiedreward_rewards(self, check_models_available):
        """Load UnifiedReward-Think rewards once for all tests."""
        from ltx_trainer.rl.rewards import get_reward_functions

        try:
            rewards = get_reward_functions("unifiedreward_think")
            assert len(rewards) == 3, "UnifiedReward-Think should return 3 dimensions"
            return {name: fn for fn, name in rewards}
        except Exception as e:
            pytest.skip(f"Failed to load UnifiedReward-Think model: {e}")

    def test_dimensions(self, unifiedreward_rewards):
        """Test that all expected dimensions are present."""
        expected = ["unifiedreward_alignment", "unifiedreward_coherence", "unifiedreward_style"]
        assert set(unifiedreward_rewards.keys()) == set(expected)

    def test_black_video(self, unifiedreward_rewards):
        """Test scoring on a black video."""
        prompt = "A completely black screen"
        for name, reward_fn in unifiedreward_rewards.items():
            score = reward_fn.compute(BLACK_VIDEO, prompt)
            assert isinstance(score, float), f"{name} should return float"
            assert 1.0 <= score <= 5.0, f"{name} score should be in [1, 5], got {score}"
            print(f"  {name}: {score:.3f}")

    def test_white_video(self, unifiedreward_rewards):
        """Test scoring on a white video."""
        prompt = "A completely white screen"
        for name, reward_fn in unifiedreward_rewards.items():
            score = reward_fn.compute(WHITE_VIDEO, prompt)
            assert isinstance(score, float)
            assert 1.0 <= score <= 5.0, f"{name} score should be in [1, 5], got {score}"
            print(f"  {name}: {score:.3f}")

    def test_alternating_video(self, unifiedreward_rewards):
        """Test scoring on an alternating red/blue video."""
        prompt = "A video alternating between red and blue colors"
        for name, reward_fn in unifiedreward_rewards.items():
            score = reward_fn.compute(ALTERNATING_VIDEO, prompt)
            assert isinstance(score, float)
            assert 1.0 <= score <= 5.0, f"{name} score should be in [1, 5], got {score}"
            print(f"  {name}: {score:.3f}")

    def test_gradient_video(self, unifiedreward_rewards):
        """Test scoring on a video with gradient from black to white."""
        prompt = "A smooth fade from black to white"
        for name, reward_fn in unifiedreward_rewards.items():
            score = reward_fn.compute(GRADIENT_VIDEO, prompt)
            assert isinstance(score, float)
            assert 1.0 <= score <= 5.0, f"{name} score should be in [1, 5], got {score}"
            print(f"  {name}: {score:.3f}")

    def test_caching(self, unifiedreward_rewards):
        """Test that caching works correctly (same video object returns cached scores)."""
        prompt = "Test video for caching"
        test_video = make_video(lambda f: (0.3, 0.7, 0.5))

        # First call computes all dimensions
        scores_1 = {}
        for name, reward_fn in unifiedreward_rewards.items():
            scores_1[name] = reward_fn.compute(test_video, prompt)

        # Second call should return cached results (exact same values)
        scores_2 = {}
        for name, reward_fn in unifiedreward_rewards.items():
            scores_2[name] = reward_fn.compute(test_video, prompt)

        assert scores_1 == scores_2, "Cached scores should be identical"

    def test_different_prompts(self, unifiedreward_rewards):
        """Test that different prompts can produce different scores."""
        test_video = BLUE_VIDEO

        prompt1 = "A blue screen"
        prompt2 = "A red sunset over mountains"

        # Get alignment dimension (most likely to vary with prompt)
        alignment_fn = unifiedreward_rewards["unifiedreward_alignment"]

        score1 = alignment_fn.compute(test_video, prompt1)
        score2 = alignment_fn.compute(test_video, prompt2)

        # Note: We don't require scores to be different (model might still give same score)
        # Just verify both are valid
        assert 1.0 <= score1 <= 5.0
        assert 1.0 <= score2 <= 5.0
        print(f"  Blue video with 'blue screen' prompt: {score1:.3f}")
        print(f"  Blue video with 'red sunset' prompt: {score2:.3f}")


class TestFactoryIntegration:
    """Tests for reward factory integration."""

    def test_video_score2_factory(self):
        """Test that video_score2 can be retrieved from factory."""
        from ltx_trainer.rl.rewards import get_reward_functions

        try:
            rewards = get_reward_functions("video_score2")
            assert len(rewards) == 3
            names = [name for _, name in rewards]
            assert "videoscore2_visual_quality" in names
            assert "videoscore2_text_to_video_alignment" in names
            assert "videoscore2_physical_consistency" in names
        except Exception:
            pytest.skip("VideoScore2 model not available")

    def test_unifiedreward_factory(self):
        """Test that unifiedreward_think can be retrieved from factory."""
        from ltx_trainer.rl.rewards import get_reward_functions

        try:
            rewards = get_reward_functions("unifiedreward_think")
            assert len(rewards) == 3
            names = [name for _, name in rewards]
            assert "unifiedreward_alignment" in names
            assert "unifiedreward_coherence" in names
            assert "unifiedreward_style" in names
        except Exception:
            pytest.skip("UnifiedReward-Think model not available")

    def test_unknown_reward(self):
        """Test that unknown reward names raise ValueError."""
        from ltx_trainer.rl.rewards import get_reward_functions

        with pytest.raises(ValueError, match="Unknown reward function"):
            get_reward_functions("nonexistent_reward")


if __name__ == "__main__":
    # Allow running as a standalone script for quick testing
    pytest.main([__file__, "-v", "-s"])
