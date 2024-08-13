using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Flame : MonoBehaviour
{
    // Start is called before the first frame update

    public GameObject flameEffects;
    public Collider flameTrigger;
    public bool onFire = false;

    void Start()
    {
        flameEffects.SetActive(onFire);
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    void OnTriggerEnter(Collider other)
    {
        if (other.gameObject.layer == LayerMask.NameToLayer("Flame"))
        {
            if (other.gameObject.GetComponent<Flame>().onFire)
            {
                onFire = true;
                flameEffects.SetActive(onFire);
                flameTrigger.enabled = onFire;
            }          
        }
    }
}
