#
# Copyright 2018 rn9dfj3
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import bpy
import numpy as np

bl_info = {
    "name": "Love2D3D",
    "author": "rn9dfj3",
    "version": (0, 1),
    "blender": (2, 79, 0),
    "location": "3D View > Object Mode > Tool Shelf > Create > Love2D3D",
    "description": "Create 3D object from 2D image",
    "warning": "",
    "support": "COMMUNITY",
    "wiki_url": "https://github.com/rn9dfj3/love2d3d/wiki",
    "tracker_url": "https://github.com/rn9dfj3/love2d3d/issues",
    "category": "Add Mesh"
}

RGBA = 4  # Color size per pixels
RGB = 3  # Color size per pixels
R = 0  # Index of color
G = 1  # Index of color
B = 2  # Index of color
A = 3  # Index of color
X = 0  # Index
Y = 1  # Index
LEFT = 2  # Index
RIGHT = 3  # Index
BOTTOM = 4  # Index
TOP = 5  # Index
QUAD = 4  # Vertex Numer of Quad
FRONT = 0
BACK = 1
NAME = "Love2D3D"  # Name of 3D object


class CreateObject(bpy.types.Operator):

    bl_idname = "object.create_love2d3d"
    bl_label = "Create love2D3D"
    bl_description = "Create 3D object from 2D image."
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        image = context.window_manager.love2d3d.image_front  # Image ID
        if image == "":
            return {"CANCELLED"}
        image = context.blend_data.images[image]  # Get image
        resolution = context.window_manager.love2d3d.rough  # Get resolution
        w, h = image.size  # Image width and height
        all = w * h
        pixels = image.pixels[:]  # Get slice of color infomation
        fronts = []
        backs = [[True for i in range(w)] for j in range(h)]
        e1 = h-resolution
        e2 = w-resolution
        opacity = context.window_manager.love2d3d.opacity
        threshold = context.window_manager.love2d3d.threshold
        for y in range(resolution, e1)[::resolution]:
            left = 0 + y * w
            b2 = RGBA * left  # Get Left color of image
            for x in range(resolution, e2)[::resolution]:
                back = False
                for v in range(resolution):
                    for u in range(resolution):
                        p = (x+u) + (y+v) * w
                        b1 = RGBA * p  # Get each color of image
                        if opacity:  # Whether opaque or not
                            c1 = pixels[b1+A]
                            c2 = pixels[b2+A]
                            back = back or c1 <= threshold
                        else:  # Whether same color or not
                            c1 = pixels[b1:b1+RGB]
                            c2 = pixels[b2:b2+RGB]
                            back = back or abs(c1[R] - c2[R]) + \
                                abs(c1[G] - c2[G]) \
                                + abs(c1[B] - c2[B]) <= threshold * 3.0
                        if back:
                            break
                    if back:
                        break
                backs[y][x] = back
                if not back:
                    fronts.append((x//resolution, y//resolution))
        del e1, e2, b1, b2, c1, c2, back, pixels, p, left
        terms = []
        for k, f in enumerate(fronts):
            fx = f[X]
            fy = f[Y]
            x = fx * resolution
            y = fy * resolution
            left = backs[y][x-resolution]
            right = backs[y][x+resolution]
            back = backs[y-resolution][x]
            top = backs[y+resolution][x]
            if not backs[y][x] and (left or right or back or top):
                terms.append((fx, fy))  # Get edge
            fronts[k] = (fx, fy, left, right, back, top)  # Insert edge info
        lens = [[0.0 for i in range(w)[::resolution]]
                for j in range(h)[::resolution]]
        if len(fronts) == 0:
            return {"CANCELLED"}
        sqAll = all ** 2
        xs = np.array([f[X] for f in fronts])  # X coordinate of each point
        ys = np.array([f[Y] for f in fronts])  # Y coordinate of each point
        ls = np.full(len(fronts), sqAll)
        for t in terms:
            ms = np.minimum(ls, np.power(t[X]-xs, 2) + np.power(t[Y] - ys, 2))
            ls = ms  # Watershed algorithm
        ms = np.sqrt(ls) + 1
        m = np.max(ms)
        ls = np.divide(ms, m)  # Nomalize
        ms = (np.sin(ls * np.pi * 0.5)+0)
        for k, f in enumerate(fronts):
            fx = f[X]
            fy = f[Y]
            ls = ms[k]/4.0  # Blur of height for edge
            lens[fy][fx] += ls
            fxi = fx+1
            fyi = fy+1
            lens[fy][fxi] += ls
            lens[fyi][fx] += ls
            lens[fyi][fxi] += ls
        del fx, fy, fxi, fyi, left, right, back, top, k, f, ms, ls, m
        verts = []
        nei = 1  # Neighbor
        uvs = []
        uvx = 0 / w
        uvy = 0 / h
        backs = []
        depth_front = context.window_manager.love2d3d.depth_front
        depth_back = context.window_manager.love2d3d.depth_back
        for f in fronts:
            x = f[X]
            y = f[Y]
            xi = x+nei
            yi = y+nei
            x1 = x * resolution
            x2 = xi * resolution
            y1 = y * resolution
            y2 = yi * resolution
            lu = x1/w
            ru = x2/w
            bu = y1/h
            tu = y2/h
            # Front face
            p1 = (x1, -lens[yi][x] * depth_front, y2)
            p2 = (x1, -lens[y][x] * depth_front, y1)
            p3 = (x2, -lens[y][xi] * depth_front, y1)
            p4 = (x2, -lens[yi][xi] * depth_front, y2)
            verts.extend([p1, p2, p3, p4])
            u1 = (lu + uvx, tu + uvy)
            u2 = (lu + uvx, bu + uvy)
            u3 = (ru + uvx, bu + uvy)
            u4 = (ru + uvx, tu + uvy)
            uvs.extend([u1, u2, u3, u4])
            backs.append(FRONT)
            # Back face
            p5 = (x2, lens[yi][xi] * depth_back, y2)
            p6 = (x2, lens[y][xi] * depth_back, y1)
            p7 = (x1, lens[y][x] * depth_back, y1)
            p8 = (x1, lens[yi][x] * depth_back, y2)
            verts.extend([p5, p6, p7, p8])
            uvs.extend([u4, u3, u2, u1])
            backs.append(BACK)
            if f[LEFT]:  # Left face
                verts.extend([p8, p7, p2, p1])
                uvs.extend([u1,u2,u2,u1])
                backs.append(FRONT)
            if f[RIGHT]:  # Right face
                vertes.extend([p4,p3,p6,p5])
                uvs.extend([u4,u3,u3,u4])
                backs.append(FRONT)
            if f[TOP]:  # Top face
                vertes.extend([p8,p1,p4,p5])
                uvs.extend([u1,u1,u4,u4])
                backs.append(FRONT)
            if f[BOTTOM]:  # Bottom face
                vertes.extend([p2,p7,p6,p3])
                uvs.extend([u2,u2,u3,u3])
                backs.append(FRONT)
        del p1, p2, p3, p4, p5, p6, p7, p8, lens, nei, x, y
        del xi, yi, lu, ru, bu, tu, x1, x2, y1, y2
        del u1, u2, u3, u4
        faces = [(0, 0, 0, 0)] * (len(verts)//QUAD)
        for n, f in enumerate(faces):
            faces[n] = (QUAD * n, QUAD * n + 1, QUAD * n + 2, QUAD * n + 3)
        msh = bpy.data.meshes.new(NAME)
        msh.from_pydata(verts, [], faces)  # Coordinate is Blender Coordinate
        msh.update()
        del verts, faces
        obj = bpy.data.objects.new(NAME, msh)  # Create 3D object
        scene = bpy.context.scene
        scene.objects.link(obj)
        bpy.ops.object.select_all(action='DESELECT')
        if bpy.ops.object.mode_set.poll():
            bpy.ops.object.mode_set(mode='OBJECT')
        obj.select = True
        bpy.context.scene.objects.active = obj
        obj.location = (-w/2, 0, -h/2)  # Translate to origin
        bpy.ops.object.transform_apply(location=True)
        scale = context.window_manager.love2d3d.scale
        obj.scale = (scale, scale, scale)
        bpy.ops.object.transform_apply(scale=True)
        channel_name = "uv"
        msh.uv_textures.new(channel_name)  # Create UV coordinate
        for idx, dat in enumerate(msh.uv_layers[channel_name].data):
            dat.uv = uvs[idx]
        del uvs, scale
        # Crate fornt material
        matf = bpy.data.materials.new('Front')
        tex = bpy.data.textures.new('Front', type='IMAGE')
        tex.image = image
        matf.texture_slots.add()
        matf.texture_slots[0].texture = tex
        obj.data.materials.append(matf)
        # Crate back material
        matb = bpy.data.materials.new('Back')
        tex = bpy.data.textures.new('Back', type='IMAGE')
        image_back = context.window_manager.love2d3d.image_back
        if image_back == "":
            tex.image = image
        else:
            image_back = context.blend_data.images[image_back]
            tex.image = image_back
        matb.texture_slots.add()
        matb.texture_slots[0].texture = tex
        obj.data.materials.append(matb)
        for k, f in enumerate(obj.data.polygons):
            f.material_index = backs[k]  # Set back material
        bpy.context.scene.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')  # Remove doubled point
        bpy.ops.mesh.remove_doubles()
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.context.scene.objects.active = obj  # Apply modifiers
        bpy.ops.object.modifier_add(type='SMOOTH')
        smo = obj.modifiers["Smooth"]
        smo.iterations = context.window_manager.love2d3d.smooth
        bpy.ops.object.modifier_add(type='DISPLACE')
        dis = obj.modifiers["Displace"]
        dis.strength = context.window_manager.love2d3d.fat
        if context.window_manager.love2d3d.modifier:
            bpy.ops.object.modifier_apply(apply_as='DATA', modifier="Smooth")
            bpy.ops.object.modifier_apply(apply_as='DATA', modifier="Displace")
        obj.select = True
        bpy.ops.object.shade_smooth()
        return {'FINISHED'}


class VIEW3D_PT_love2d3d(bpy.types.Panel):

    bl_label = "Love2D3D"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'TOOLS'
    bl_category = "Create"
    bl_context = "objectmode"

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        col.label(text="Object", icon="OBJECT_DATA")
        col.operator(CreateObject.bl_idname, text="Create")
        col = layout.column(align=True)
        col.label(text="Image", icon="IMAGE_DATA")
        col.operator("image.open", icon="FILESEL")
        col.prop_search(context.window_manager.love2d3d,
                        "image_front", context.blend_data, "images")
        col.prop_search(context.window_manager.love2d3d,
                        "image_back", context.blend_data, "images")
        layout.separator()
        col = layout.column(align=True)
        col.label(text="Separation", icon="IMAGE_RGB_ALPHA")
        col.prop(context.window_manager.love2d3d, "threshold")
        col.prop(context.window_manager.love2d3d, "opacity")
        layout.separator()
        col = layout.column(align=True)
        col.label(text="Geometry", icon="EDITMODE_HLT")
        col.prop(context.window_manager.love2d3d, "depth_front")
        col.prop(context.window_manager.love2d3d, "depth_back")
        col.prop(context.window_manager.love2d3d, "scale")
        layout.separator()
        col = layout.column(align=True)
        col.label(text="Quality", icon="MOD_SMOOTH")
        col.prop(context.window_manager.love2d3d, "rough")
        col.prop(context.window_manager.love2d3d, "smooth")
        col.prop(context.window_manager.love2d3d, "fat")
        layout.separator()
        col = layout.column(align=True)
        col.label(text="Option", icon="SCRIPTWIN")
        col.prop(context.window_manager.love2d3d, "modifier")


class Love2D3DProps(bpy.types.PropertyGroup):
    image_front = bpy.props.StringProperty(name="Front Image",
                                           description="Front image of mesh")
    image_back = bpy.props.StringProperty(name="Back Image",
                                          description="Back image of mesh")
    rough = bpy.props.IntProperty(name="Rough",
                                  description="Roughness of image", min=1,
                                  default=8, subtype="PIXEL")
    smooth = bpy.props.IntProperty(name="Smooth",
                                   description="Smoothness of mesh",
                                   min=1, default=30)
    scale = bpy.props.FloatProperty(name="Scale",
                                    description="Length per pixel",
                                    unit="LENGTH", min=0.01, default=0.01)
    depth_front = bpy.props.FloatProperty(name="Front Depth",
                                          description="Depth of front face",
                                          unit="LENGTH", min=0, default=40)
    depth_back = bpy.props.FloatProperty(name="Back Depth",
                                         description="Depth of back face",
                                         unit="LENGTH", min=0, default=40)
    fat = bpy.props.FloatProperty(name="Fat",
                                  description="Fat of mesh",
                                  default=0.2, min=0.0)
    modifier = bpy.props.BoolProperty(name="Modifier",
                                      description="Apply modifiers to object",
                                      default=True)
    threshold = bpy.props.FloatProperty(name="Threshold",
                                        description="Threshold of image",
                                        min=0.0, max=1.0,
                                        default=0.0, subtype="FACTOR")
    opacity = bpy.props.BoolProperty(name="Opacity",
                                     description="Use Opacity for threshold")


def register():
    bpy.utils.register_module(__name__)
    bpy.types.WindowManager.love2d3d \
        = bpy.props.PointerProperty(type=Love2D3DProps)


def unregister():
    del bpy.types.WindowManager.love2d3d
    bpy.utils.unregister_module(__name__)


if __name__ == "__main__":
    register()
