using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Item : MonoBehaviour
{
    // Start is called before the first frame update
    void Start() { }

    // Update is called once per frame
    void Update() {}

    private void OnTriggerStay(Collider other)
    {
        if (other.tag == "Player" &&  Input.GetKey(KeyCode.E))
        {
            Use();           
        }
    }

    virtual public void Use() { }
}
