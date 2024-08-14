using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ResetFood : MonoBehaviour
{

    public GameObject[] prefabsToSpawn;  // Assign your prefabs in the Unity editor
    public Transform spawnPoint;         // Set the position where the prefabs will be spawned

    // Start is called before the first frame update
    void Start()
    {

    }

    public void SpawnAllPrefabs()
    {
        if (prefabsToSpawn.Length == 0)
        {
            Debug.LogError("No prefabs assigned to the SpawnManager.");
            return;
        }

        foreach (GameObject prefab in prefabsToSpawn)
        {
            // Instantiate each prefab at the specified position and rotation
            Instantiate(prefab, spawnPoint.position, spawnPoint.rotation);
        }
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
