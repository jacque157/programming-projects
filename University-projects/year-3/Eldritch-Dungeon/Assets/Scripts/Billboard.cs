using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Billboard : MonoBehaviour
{
    private Camera mainCamera;

    // Start is called before the first frame update
    void Start()
    {
        mainCamera = Camera.main;
    }

    // Update is called once per frame
    void Update()
    {
        transform.rotation = Quaternion.Euler(mainCamera.transform.rotation.eulerAngles.x, mainCamera.transform.rotation.eulerAngles.y, mainCamera.transform.rotation.eulerAngles.z);
    }
}
