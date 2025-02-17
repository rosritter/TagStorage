import requests
import unittest

class TestChromaAPI(unittest.TestCase):
    def setUp(self):
        self.base_url = "http://localhost:8001"
        self.test_collection = "test_collection"
        
    def tearDown(self):
        # Clean up by deleting test collection
        try:
            requests.delete(f"{self.base_url}/collections/{self.test_collection}")
        except:
            pass

    def test_create_collection(self):
        # Test creating a collection
        response = requests.post(
            f"{self.base_url}/collections/",
            json={"name": self.test_collection, "metadata": {"description": "test collection"}}
        )
        self.assertEqual(response.status_code, 200)
        
    def test_add_and_query_items(self):
        # Create collection
        requests.post(
            f"{self.base_url}/collections/",
            json={"name": self.test_collection}
        )
        
        # Test data
        texts = [
            "The quick brown fox jumps over the lazy dog",
            "A quick brown dog jumps over the lazy fox"
        ]
        
        # Add items
        response = requests.post(
            f"{self.base_url}/collections/{self.test_collection}/add",
            json={
                "texts": texts,
                "ids": ["text1", "text2"],
                "metadatas": [{"source": "test1"}, {"source": "test2"}]
            }
        )
        self.assertEqual(response.status_code, 200)
        
        # Query items
        query_response = requests.post(
            f"{self.base_url}/collections/{self.test_collection}/query",
            json={
                "texts": ["quick brown fox"],
                "n_results": 2
            }
        )
        self.assertEqual(query_response.status_code, 200)
        results = query_response.json()
        
        # Check if scores are included
        self.assertIn('scores', results)
        self.assertTrue(len(results['scores']) > 0)
        
    def test_delete_items(self):
        # Create collection and add items
        requests.post(
            f"{self.base_url}/collections/",
            json={"name": self.test_collection}
        )
        
        # Add items
        requests.post(
            f"{self.base_url}/collections/{self.test_collection}/add",
            json={
                "texts": ["test text"],
                "ids": ["test_id"],
                "metadatas": [{"source": "test"}]
            }
        )
        
        # Delete items
        response = requests.delete(
            f"{self.base_url}/collections/{self.test_collection}/delete",
            json=["test_id"]
        )
        self.assertEqual(response.status_code, 200)
        
    def test_list_collections(self):
        # Create a collection first
        requests.post(
            f"{self.base_url}/collections/",
            json={"name": self.test_collection}
        )
        
        # Test listing collections
        response = requests.get(f"{self.base_url}/collections")
        self.assertEqual(response.status_code, 200)
        collections = response.json()
        self.assertIn('collections', collections)
        
    def test_get_collection_info(self):
        # Create a collection first
        requests.post(
            f"{self.base_url}/collections/",
            json={"name": self.test_collection}
        )
        
        # Test getting collection info
        response = requests.get(f"{self.base_url}/collections/{self.test_collection}")
        self.assertEqual(response.status_code, 200)
        info = response.json()
        self.assertEqual(info['name'], self.test_collection)

if __name__ == '__main__':
    unittest.main()