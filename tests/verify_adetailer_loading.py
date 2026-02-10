import sys
import os
from unittest.mock import MagicMock, patch
import unittest
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from posts.infrastructure.adetailer import ADetailer
from posts.domain.value_objects import DeviceConfig

class TestADetailerLoading(unittest.TestCase):
    def setUp(self):
        # Mock DeviceConfig
        self.device_config = MagicMock(spec=DeviceConfig)
        self.device_config.device = 'cpu'
        self.adetailer = ADetailer(self.device_config)

    @patch('posts.infrastructure.adetailer.hf_hub_download')
    @patch('posts.infrastructure.adetailer.YOLO')
    def test_load_local_success(self, mock_yolo, mock_hf_hub_download):
        """Test that if local files exist, only one call to hf_hub_download is made with local_files_only=True"""
        print("\nRunning test_load_local_success...")
        
        # Setup mock to return a path when called with local_files_only=True
        def mock_download(repo_id, filename, **kwargs):
            if kwargs.get('local_files_only'):
                return "/tmp/mock_model_path"
            return None
            
        mock_hf_hub_download.side_effect = mock_download
        
        model = self.adetailer._load_yolo_model("test_model.pt", "test/repo")
        
        # Verify hf_hub_download was called once with local_files_only=True
        mock_hf_hub_download.assert_called_once_with(repo_id="test/repo", filename="test_model.pt", local_files_only=True)
        print("Success: Checked local cache only.")
        
        # Verify YOLO was initialized with the path
        mock_yolo.assert_called_once_with("/tmp/mock_model_path")
        
        self.assertIsNotNone(model)

    @patch('posts.infrastructure.adetailer.hf_hub_download')
    @patch('posts.infrastructure.adetailer.YOLO')
    def test_load_fallback_download(self, mock_yolo, mock_hf_hub_download):
        """Test that if local load fails, it falls back to standard download"""
        print("\nRunning test_load_fallback_download...")
        
        # Setup mock to raise Exception for first call (local), then return path for second call
        def mock_download(repo_id, filename, **kwargs):
            if kwargs.get('local_files_only'):
                raise Exception("Not found locally")
            return "/tmp/downloaded_model_path"
            
        mock_hf_hub_download.side_effect = mock_download
        
        model = self.adetailer._load_yolo_model("test_model.pt", "test/repo")
        
        # Verify hf_hub_download was called twice
        self.assertEqual(mock_hf_hub_download.call_count, 2)
        print("Success: Attempted local load, failed, and fell back to download.")
        
        # Check calls
        calls = mock_hf_hub_download.call_args_list
        # First call has local_files_only=True
        self.assertTrue(calls[0].kwargs.get('local_files_only'))
        # Second call does not (or has it False/None)
        self.assertFalse(calls[1].kwargs.get('local_files_only'))
        
        # Verify YOLO was initialized with the downloaded path
        mock_yolo.assert_called_once_with("/tmp/downloaded_model_path")

if __name__ == '__main__':
    unittest.main()
