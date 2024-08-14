using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TiltFallCollider : MonoBehaviour
{
    public TiltPuzzle tiltPuzzle;
    public GameObject key;
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Respawn"))
        {
            tiltPuzzle.Respawn();
        }
        else if(other.CompareTag("Finish"))
        {
            key.transform.SetParent(null);
            tiltPuzzle.Break(key);

        }
    }
}
