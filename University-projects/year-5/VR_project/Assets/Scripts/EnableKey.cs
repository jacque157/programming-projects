using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EnableKey : MonoBehaviour
{
    // Start is called before the first frame update
    private GameObject key;
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }
    public void EnableRB()
    {
        key.GetComponent<Rigidbody>().isKinematic = false;
    }
}
