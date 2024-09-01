using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MouseLook : MonoBehaviour
{
    public float mouseSensitivity = 500f;
    public Transform playerBody;

    float xRotation = 0f;
    float yRotation = 0f;

    // Start is called before the first frame update
    void Start()
    {
        Cursor.lockState = CursorLockMode.Locked;
    }

    // Update is called once per frame
    void Update()
    {
        float mouseX = Input.GetAxis("Mouse X") * mouseSensitivity * Time.deltaTime;
        float mouseY = Input.GetAxis("Mouse Y") * mouseSensitivity * Time.deltaTime;

        xRotation = Mathf.Clamp(xRotation - mouseY, -90f, 90f);
        yRotation = (yRotation + mouseX) % 360;

        transform.localRotation = Quaternion.Euler(xRotation, yRotation, 0f);

        
    }

    
}
