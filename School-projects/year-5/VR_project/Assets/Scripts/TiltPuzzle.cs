using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR.Interaction.Toolkit;

public class TiltPuzzle : MonoBehaviour
{
    public Rigidbody rb;
    public Rigidbody table;
    public ConfigurableJoint cj;

    public GameObject sphere;
    public GameObject sphereRespawnPos;
    public GameObject restartText;
    
    public int handsHolding = 0;
    public float timeToRespawn = 2;

    private Rigidbody sphereRb;

    public bool finished = false;

    // Start is called before the first frame update
    void Start()
    {
        table.isKinematic = true;
        cj.connectedBody = null;
        restartText.SetActive(false);    
        handsHolding = 0;

        sphereRb = sphere.GetComponent<Rigidbody>();

    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public void OnSelect()
    {
        handsHolding += 1;
        cj.connectedBody = table;

        table.isKinematic = false;

    }

    public void OnDeselect()
    {
        handsHolding -= 1;

        if (handsHolding==0)
        {
            table.isKinematic = true;

        }

    }

    public void Respawn()
    {
        if (!finished)
            StartCoroutine(RespawnCo());
    }

    public IEnumerator RespawnCo()
    {
        
        restartText.SetActive(true);
        sphere.SetActive(false);
        sphereRb.velocity = Vector3.zero;
        sphereRb.isKinematic = true;

        sphere.transform.position = sphereRespawnPos.transform.position;
        
        yield return new WaitForSeconds(timeToRespawn);
        
        sphereRb.isKinematic = false;
        sphere.SetActive(true);
        restartText.SetActive(false);



    }

    public void Break(GameObject key)
    {
        if (finished)
            return;
        key.GetComponentInChildren<BoxCollider>().enabled = true;
        key.GetComponentInChildren<Rigidbody>().isKinematic = false;
        key.GetComponentInChildren<XRGrabInteractable>().enabled = true;
        finished = true;
        Destroy(sphere,1.0f);

    }
}
