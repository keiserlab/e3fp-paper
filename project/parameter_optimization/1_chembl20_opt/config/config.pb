language: PYTHON
name:     "wrapper"

variable {
 name: "level"
 type: INT
 size: 1
 min:  2
 max:  5
}

variable {
 name: "radius_multiplier"
 type: FLOAT
 size: 1
 min:  1.3
 max:  2.8
}

variable {
 name: "bits"
 type: ENUM
 size: 1
 options: "1024"
}

variable {
 name: "first"
 type: INT
 size: 1
 min:  1
 max:  35
}

variable {
 name: "conformers"
 type: ENUM
 size: 1
 options: "conformers_proto_rms0.5"
}


