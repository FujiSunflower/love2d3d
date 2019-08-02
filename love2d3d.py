#
# Copyright 2018 Fuji Sunflower
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
from bpy.props import FloatProperty, BoolProperty, StringProperty, IntProperty
from bpy_extras.object_utils import AddObjectHelper, object_data_add
import bgl
from mathutils import Vector, Matrix
from mathutils.kdtree import KDTree
import gpu
from gpu_extras.batch import batch_for_shader
import sys

bl_info = {
    "name": "Love2D3D",
    "author": "Fuji Sunflower",
    "version": (3, 0),
    "blender": (2, 80, 0),
    "location": "3D View > Menu > Add > Mesh, Armature. Sidebar > View",
    "description": "Add 3D object from image or its armature.",
    "warning": "",
    "support": "COMMUNITY",
    "wiki_url": "https://github.com/FujiSunflower/love2d3d/wiki",
    "tracker_url": "https://github.com/FujiSunflower/love2d3d/issues",
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
BOUND_LEFT = 0  # Index of bounds
BOUND_RIGHT = 1  # Index of bounds
BOUND_BACK = 2  # Index of bounds
BOUND_FRONT = 3  # Index of bounds
BOUND_TOP = 4  # Index of bounds
BOUND_BOTTOM = 5  # Index of bounds
BOUND_CENTER = 6  # Index of bounds
BRANCH_BOOST = 3.0
BONE_TYPE_ANY = -1  # Index of bone types
BONE_TYPE_BODY = 0  # Index of bone types
BONE_TYPE_HEAD = 1  # Index of bone types
BONE_TYPE_ARM_LEFT = 2  # Index of bone types
BONE_TYPE_ARM_RIGHT = 3  # Index of bone types
BONE_TYPE_LEG_LEFT = 4  # Index of bone types
BONE_TYPE_LEG_RIGHT = 5  # Index of bone types
BONE_TYPE_FINGER_LEFT = 7  # Index of bone types
BONE_TYPE_FINGER_RIGHT = 8  # Index of bone types


class LOVE2D3D_OT_preview(bpy.types.Operator):

    bl_idname = "object.love2d3d_preview_mesh"
    bl_label = "Preview love2D3D mesh"
    bl_description = "Preview mesh of love2D3D"
    bl_options = {'INTERNAL'}
    _handle = None
    vertex_shader = '''
        uniform mat4 modelMatrix;
        uniform mat4 viewProjectionMatrix;

        in vec2 position;
        in vec2 uv;

        out vec2 uvInterp;

        void main()
        {
            uvInterp = uv;
            vec4 p = vec4(position, 0.0, 1.0);
            gl_Position = viewProjectionMatrix * modelMatrix * p;
        }
    '''

    fragment_shader = '''
        uniform sampler2D image;

        in vec2 uvInterp;

        void main()
        {
            gl_FragColor = texture(image, uvInterp);
            gl_FragColor.a = 0.5;
        }
    '''

    def modal(self, context, event):
        area = context.area
        if area is None:
            return {'PASS_THROUGH'}
        area.tag_redraw()
        preview = context.window_manager.love2d3d.preview
        if not preview:
            return {'FINISHED'}
        return {'PASS_THROUGH'}

    def preview(self, context):
        image_front = context.window_manager.love2d3d.image_front  # Image ID
        if image_front == "" or image_front is None:
            return
        self.shader = gpu.types.GPUShader(
            self.vertex_shader, self.fragment_shader)
        scale = context.window_manager.love2d3d.scale
        self.image = context.blend_data.images[image_front]  # Get image
        if self.image.gl_load():
            raise Exception()
        w, h = self.image.size
        w *= scale
        h *= scale
        view_align = context.window_manager.love2d3d.view_align
        lb = (-w / 2.0, -h / 2.0)
        rb = (w / 2.0, -h / 2.0)
        rt = (w / 2.0, h / 2.0)
        lt = (-w / 2.0, h / 2.0)
        self.batch = batch_for_shader(
            self.shader, 'TRI_FAN',
            {
                "position": (lb, rb, rt, lt),
                "uv": ((0, 0), (1, 0), (1, 1), (0, 1)),
            },
        )
        bgl.glActiveTexture(bgl.GL_TEXTURE0)
        bgl.glBindTexture(bgl.GL_TEXTURE_2D, self.image.bindcode)
        cursor = context.scene.cursor.location
        bgl.glEnable(bgl.GL_BLEND)
        self.shader.bind()
        if view_align:
            iview = Matrix(context.region_data.view_matrix). \
                inverted_safe().to_3x3().to_4x4()
            self.shader.uniform_float(
                "modelMatrix", Matrix.Translation(cursor) @ iview)
        else:
            mat_rot = Matrix.Rotation(np.radians(90.0), 4, 'X')  # rot matrix
            self.shader.uniform_float(
                "modelMatrix", Matrix.Translation(cursor) @ mat_rot)
        self.shader.uniform_float(
            "viewProjectionMatrix", context.region_data.perspective_matrix)
        self.shader.uniform_float("image", 0)
        self.batch.draw(self.shader)
        bgl.glDisable(bgl.GL_BLEND)

    def _handle_remove(self, context):
        if LOVE2D3D_OT_preview._handle is not None:
            bpy.types.SpaceView3D.draw_handler_remove(
                LOVE2D3D_OT_preview._handle, 'WINDOW')
            LOVE2D3D_OT_preview._handle = None

    def _handle_add(self, context):
        if LOVE2D3D_OT_preview._handle is None:
            space = bpy.types.SpaceView3D
            LOVE2D3D_OT_preview._handle = space.draw_handler_add(
                self.preview, (context,), 'WINDOW', 'POST_VIEW')

    def invoke(self, context, event):
        preview = context.window_manager.love2d3d.preview
        if context.area.type == 'VIEW_3D':
            if not preview:
                context.window_manager.love2d3d.preview = True
            else:
                context.window_manager.love2d3d.preview = False
                self._handle_remove(context)
                return {'FINISHED'}
            self._handle_add(context)
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            self.report({'WARNING'}, "View3D not found, cannot run operator")
            self._handle_remove(context)
            return {'CANCELLED'}


class LOVE2D3D_OT_createObject(bpy.types.Operator, AddObjectHelper):

    bl_idname = "object.love2d3d_add_mesh"
    bl_label = "Add love2D3D Mesh"
    bl_description = "Add 3D object from 2D image"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        love2d3d = context.window_manager.love2d3d
        # debug_time = datetime.datetime.today()
        image = love2d3d.image_front  # Image ID
        if image == "":
            return {"CANCELLED"}
        image = context.blend_data.images[image]  # Get image
        resolution = love2d3d.rough  # Get resolution
        w, h = image.size  # Image width and height
        pixels = image.pixels[:]  # Get slice of color infomation
        fronts = []
        backs = [[True for i in range(w)] for j in range(h)]  # whether bg
        ex = h - resolution  # End of list
        ey = w - resolution  # End of list
        opacity = love2d3d.opacity  # use opacity or not
        threshold = love2d3d.threshold  # bg's threshold
        for y in range(resolution, ex)[::resolution]:
            left = 0 + y * w
            il = RGBA * left  # Get left index of color in image
            for x in range(resolution, ey)[::resolution]:
                back = False
                for v in range(resolution):
                    for u in range(resolution):
                        p = (x + u) + (y + v) * w  # cuurent index in pixels
                        i = RGBA * p  # Get each index of color in image
                        if opacity:  # Whether opaque or not
                            c = pixels[i + A]  # each opacity in image
                            cl = pixels[il + A]  # left opacity in image
                            back = back or c <= threshold
                        else:  # Whether same color or not
                            c = pixels[i:i + RGB]  # each RGB in image
                            cl = pixels[il:il + RGB]  # left RGB in image
                            back = back or abs(c[R] - cl[R]) + \
                                abs(c[G] - cl[G]) \
                                + abs(cl[B] - cl[B]) <= threshold * 3.0
                        if back:
                            break
                    if back:
                        break
                backs[y][x] = back
                if not back:
                    fronts.append((x // resolution, y // resolution))
        del ex, ey, i, il, c, cl, back, pixels, p, left
        terms = []  # Edges of image
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
        kd = KDTree(len(terms))
        for i, t in enumerate(terms):
            kd.insert((t[X], t[Y], 0), i)
        kd.balance()
        ls = [0.0 for f in fronts]
        for k, f in enumerate(fronts):
            co_find = (f[X], f[Y], 0)
            co, index, dist = kd.find(co_find)
            ls[k] = dist
        # ms = np.sqrt(ls) + 1 # length array with softning
        ms = np.array([l + 1 for l in ls])
        m = np.max(ms)
        ls = np.divide(ms, m)  # Nomalize
        ms = (np.sin(ls * np.pi * 0.5) + 0)
        # ms = (np.arcsin(ls) + 0)

        for k, f in enumerate(fronts):
            fx = f[X]
            fy = f[Y]
            ls = ms[k] / 4.0  # Blur of height for edge
            lens[fy][fx] += ls
            fxi = fx + 1
            fyi = fy + 1
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
        view_align = love2d3d.view_align
        scale = love2d3d.scale
        s = min(w, h) / 8
        depth_front = s * love2d3d.depth_front * scale
        depth_back = s * love2d3d.depth_back * scale
        for f in fronts:
            x = f[X]
            y = f[Y]
            xi = x + nei
            yi = y + nei
            x1 = x * resolution
            x2 = xi * resolution
            y1 = y * resolution
            y2 = yi * resolution
            lu = x1 / w
            ru = x2 / w
            bu = y1 / h
            tu = y2 / h
            x1 = (x1 - w / 2) * scale
            x2 = (x2 - w / 2) * scale
            y1 = (y1 - h / 2) * scale
            y2 = (y2 - h / 2) * scale
            # Front face
            if view_align:
                p1 = (x1, y2, lens[yi][x] * depth_front)
                p2 = (x1, y1, lens[y][x] * depth_front)
                p3 = (x2, y1, lens[y][xi] * depth_front)
                p4 = (x2, y2, lens[yi][xi] * depth_front)
            else:
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
            if view_align:
                p5 = (x2,  y2, -lens[yi][xi] * depth_back)
                p6 = (x2, y1, -lens[y][xi] * depth_back)
                p7 = (x1, y1, -lens[y][x] * depth_back)
                p8 = (x1, y2, -lens[yi][x] * depth_back)
            else:
                p5 = (x2, lens[yi][xi] * depth_back, y2)
                p6 = (x2, lens[y][xi] * depth_back, y1)
                p7 = (x1, lens[y][x] * depth_back, y1)
                p8 = (x1, lens[yi][x] * depth_back, y2)
            verts.extend([p5, p6, p7, p8])
            uvs.extend([u4, u3, u2, u1])
            backs.append(BACK)
            if f[LEFT]:  # Left face
                verts.extend([p8, p7, p2, p1])
                uvs.extend([u1, u2, u2, u1])
                backs.append(FRONT)
            if f[RIGHT]:  # Right face
                verts.extend([p4, p3, p6, p5])
                uvs.extend([u4, u3, u3, u4])
                backs.append(FRONT)
            if f[TOP]:  # Top face
                verts.extend([p8, p1, p4, p5])
                uvs.extend([u1, u1, u4, u4])
                backs.append(FRONT)
            if f[BOTTOM]:  # Bottom face
                verts.extend([p2, p7, p6, p3])
                uvs.extend([u2, u2, u3, u3])
                backs.append(FRONT)
        del p1, p2, p3, p4, p5, p6, p7, p8, lens, nei, x, y
        del xi, yi, lu, ru, bu, tu, x1, x2, y1, y2
        del u1, u2, u3, u4
        faces = [(0, 0, 0, 0)] * (len(verts) // QUAD)
        for n, f in enumerate(faces):
            faces[n] = (QUAD * n, QUAD * n + 1, QUAD * n + 2, QUAD * n + 3)
        msh = bpy.data.meshes.new(NAME)
        msh.from_pydata(verts, [], faces)  # Coordinate is Blender Coordinate
        msh.update()
        del verts, faces
        # obj = object_data_add(context, msh, operator=self).object
        obj = object_data_add(context, msh, operator=self)
        if view_align:
            obj.rotation_euler = Matrix(context.region_data.view_matrix).\
                inverted_safe().to_euler()
        context.view_layer.objects.active = obj
        channel_name = "uv"
        msh.uv_layers.new(name=channel_name)  # Create UV coordinate
        for idx, dat in enumerate(msh.uv_layers[channel_name].data):
            dat.uv = uvs[idx]
        del uvs
        matf = bpy.data.materials.new('Front')  # Crate fornt material
        matf.use_nodes = True
        shader = matf.node_tree.nodes["Principled BSDF"]
        node = matf.node_tree.nodes.new("ShaderNodeTexImage")
        node.image = image
        input = node.outputs[0]
        output = matf.node_tree.get_output_node("ALL").inputs[0]
        matf.node_tree.links.new(input, shader.inputs[0])
        matf.node_tree.links.new(shader.outputs[0], output)
        obj.data.materials.append(matf)
        matb = bpy.data.materials.new('Back')  # Crate back material
        matb.use_nodes = True
        shader = matb.node_tree.nodes["Principled BSDF"]
        node = matb.node_tree.nodes.new("ShaderNodeTexImage")
        image_back = love2d3d.image_back
        if image_back == "":
            node.image = image
        else:
            image = context.blend_data.images[image_back]
            node.image = image
        input = node.outputs[0]
        output = matb.node_tree.get_output_node("ALL").inputs[0]
        matb.node_tree.links.new(input, shader.inputs[0])
        matb.node_tree.links.new(shader.outputs[0], output)
        obj.data.materials.append(matb)
        for k, f in enumerate(obj.data.polygons):
            f.material_index = backs[k]  # Set back material
        context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')  # Remove doubled point
        bpy.ops.mesh.remove_doubles()
        bpy.ops.object.mode_set(mode='OBJECT')
        context.view_layer.objects.active = obj
        bpy.ops.object.modifier_add(type='SMOOTH')
        smo = obj.modifiers["Smooth"]
        smo.iterations = love2d3d.smooth
        bpy.ops.object.modifier_add(type='DISPLACE')
        dis = obj.modifiers["Displace"]
        dis.strength = love2d3d.fat * scale / 0.01
        dec = None
        if love2d3d.decimate:
            bpy.ops.object.modifier_add(type='DECIMATE')
            dec = obj.modifiers["Decimate"]
            dec.ratio = love2d3d.decimate_ratio
        if love2d3d.modifier:
            bpy.ops.object.modifier_apply(apply_as='DATA', modifier="Smooth")
            bpy.ops.object.modifier_apply(apply_as='DATA', modifier="Displace")
            if love2d3d.decimate:
                bpy.ops.object.modifier_apply(
                    apply_as='DATA', modifier="Decimate")
        obj.select_set(True)
        bpy.ops.object.shade_smooth()
        return {'FINISHED'}

    def draw(self, context):
        layout = self.layout
        love2d3d = context.window_manager.love2d3d
        col = layout.column(align=True)
        row = col.row()
        row.label(text="Image", icon="IMAGE_DATA")
        row.operator("image.open", icon="FILEBROWSER", text="")
        row.operator("image.new", icon="DUPLICATE", text="")
        col.prop_search(love2d3d,
                        "image_front", context.blend_data, "images")
        col.prop_search(love2d3d,
                        "image_back", context.blend_data, "images")
        layout.separator()
        col = layout.column(align=True)
        col.label(text="Separation", icon="IMAGE_RGB_ALPHA")
        col.prop(love2d3d, "threshold")
        col.prop(love2d3d, "opacity")
        layout.separator()
        col = layout.column(align=True)
        col.label(text="Geometry", icon="EDITMODE_HLT")
        col.prop(love2d3d, "view_align")
        col.prop(love2d3d, "depth_front")
        col.prop(love2d3d, "depth_back")
        col.prop(love2d3d, "scale")
        layout.separator()
        col = layout.column(align=True)
        col.label(text="Quality", icon="MOD_SMOOTH")
        col.prop(love2d3d, "rough")
        col.prop(love2d3d, "smooth")
        col.prop(love2d3d, "fat")
        layout.separator()
        col = layout.column(align=True)
        col.label(text="Decimate", icon="MOD_DECIM")
        col.prop(love2d3d, "decimate")
        col.prop(love2d3d, "decimate_ratio")
        layout.separator()
        col = layout.column(align=True)
        col.label(text="Option", icon="MODIFIER")
        col.prop(love2d3d, "modifier")

    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_props_dialog(self)


class LOVE2D3D_OT_createArmature(bpy.types.Operator):

    bl_idname = "object.love2d3d_add_aramature"
    bl_label = "Add love2d3d armature"
    bl_description = "Add armature to selected objects"
    bl_options = {'REGISTER', 'UNDO'}
    finger_limit_angle: bpy.props.FloatProperty(
        name="Finger limit", description="Limit angle of finger",
        min=0.0, default=np.radians(8.0), subtype='ANGLE')
    hips_limit_angle: bpy.props.FloatProperty(
        name="Hips limit", description="Limit angle of hips",
        min=0.0, max=np.radians(90.0),
        default=np.radians(5.0), subtype='ANGLE')
    center_limit_angle: bpy.props.FloatProperty(
        name="Center limit", description="Limit angle of center",
        min=0.0, max=np.radians(90.0),
        default=np.radians(5.0), subtype='ANGLE')
    arm_limit_angle: bpy.props.FloatProperty(
        name="Arm limit", description="Limit angle of arm",
        min=0.0, max=np.radians(90.0),
        default=np.radians(30.0), subtype='ANGLE')
    leg_limit_angle: bpy.props.FloatProperty(
        name="Leg limit", description="Limit angle of leg",
        min=0.0, max=np.radians(90.0),
        default=np.radians(5.0), subtype='ANGLE')
    any_limit_angle: bpy.props.FloatProperty(
        name="Any limit", description="Limit angle of any bone",
        min=0.0, max=np.radians(90.0),
        default=np.radians(30.0), subtype='ANGLE')
    hand_limit_angle: bpy.props.FloatProperty(
        name="Hand limit", description="Limit angle of hand",
        min=0.0, max=np.radians(90.0),
        default=np.radians(45.0), subtype='ANGLE')
    branch_boost: bpy.props.FloatProperty(
        name="Boost", description="How many points hit as branch",
        min=0.01, default=3.0)
    finger_branch_boost: bpy.props.FloatProperty(
        name="Finger boost",
        description="How many points hit as branch in finger",
        min=0.01, default=3.0)
    gather_ratio: bpy.props.FloatProperty(
        name="Gather", description="How many branchs gather",
        min=0.0, max=100.0, default=10, subtype='PERCENTAGE')
    finger_gather_ratio: bpy.props.FloatProperty(
        name="Finger gather", description="How many branchs gather in finger",
        min=0.0, max=100.0, default=1, subtype='PERCENTAGE')
    tip_gather_ratio: bpy.props.FloatProperty(
        name="Tip gather", description="How many tips gather in finger",
        min=0.0, max=100.0, default=3, subtype='PERCENTAGE')

    def execute(self, context):
        return self.skinning(context)

    def draw(self, context):
        love2d3d = context.window_manager.love2d3d
        layout = self.layout
        layout.label(text="Main", icon="ARMATURE_DATA")
        col = layout.column(align=True)
        col.label(text="Resolution", icon="LATTICE_DATA")
        col.prop(love2d3d, "armature_resolution")
        col = layout.column(align=True)
        col.label(text="Limit", icon="CONSTRAINT")
        col.prop(self, "hips_limit_angle")
        col.prop(self, "center_limit_angle")
        col.prop(self, "arm_limit_angle")
        col.prop(self, "leg_limit_angle")
        col.prop(self, "any_limit_angle")
        col = layout.column(align=True)
        col.label(text="Amount", icon="EDITMODE_HLT")
        col.prop(self, "branch_boost")
        col.prop(self, "gather_ratio")
        layout.separator()
        # layout = layout.column(align=True)
        layout.label(text="Finger", icon="HAND")
        col = layout.column(align=True)
        col.prop(love2d3d, "armature_finger")
        col.label(text="Resolution", icon="LATTICE_DATA")
        col.prop(love2d3d, "armature_finger_resolution")
        col = layout.column(align=True)
        col.label(text="Limit", icon="CONSTRAINT")
        col.prop(self, "hand_limit_angle")
        col.prop(self, "finger_limit_angle")
        col = layout.column(align=True)
        col.label(text="Amount", icon="EDITMODE_HLT")
        col.prop(self, "finger_branch_boost")
        col.prop(self, "finger_gather_ratio")
        col.prop(self, "tip_gather_ratio")

    def bound_loc(self, obj):
        """
            Getting bounds of object.
        """
        bound = obj.bound_box
        mat = Matrix(obj.matrix_world)
        xs = []
        ys = []
        zs = []
        for b in bound:
            loc = mat @ Vector(b)
            xs.append(loc.x)
            ys.append(loc.y)
            zs.append(loc.z)
        left = max(xs)
        right = min(xs)
        back = max(ys)
        front = min(ys)
        top = max(zs)
        bottom = min(zs)
        center = Vector(((left + right) * 0.5, (back + front) * 0.5,
                        (top + bottom) * 0.5))
        return (left, right, back, front, top, bottom, center)

    def primary_obj(self, group):
        """
            Deciding of primary bone in group.
        """
        max_volume = 0.0
        max_obj = None
        for obj in group:
            b = self.bound_loc(obj)
            le = b[BOUND_LEFT]
            ri = b[BOUND_RIGHT]
            ba = b[BOUND_BACK]
            fr = b[BOUND_FRONT]
            to = b[BOUND_TOP]
            bo = b[BOUND_BOTTOM]
            volume = (le - ri) * (ba - fr) * (to - bo)
            if max_volume < volume:
                max_volume = volume
                max_obj = obj
        return max_obj

    def _make_group(self, objects, index, hits):
        """
            Recursion call of objects collision.
        """
        current_count = len(hits)
        b = self.bound_loc(objects[index])
        le = b[BOUND_LEFT]
        ri = b[BOUND_RIGHT]
        ba = b[BOUND_BACK]
        fr = b[BOUND_FRONT]
        to = b[BOUND_TOP]
        bo = b[BOUND_BOTTOM]
        neighbors = []
        for k, neighbor in enumerate(objects):
            if index == k:
                continue
            n = self.bound_loc(neighbor)
            n_le = n[BOUND_LEFT]
            n_ri = n[BOUND_RIGHT]
            n_ba = n[BOUND_BACK]
            n_fr = n[BOUND_FRONT]
            n_to = n[BOUND_TOP]
            n_bo = n[BOUND_BOTTOM]
            avoid_x = le < n_ri or n_le < ri
            avoid_y = ba < n_fr or n_ba < fr
            avoid_z = to < n_bo or n_to < bo
            avoid = avoid_x or avoid_y or avoid_z
            if not avoid:  # Hit
                neighbors.append(k)
        for neighbor in neighbors:
            already = False
            for hit in hits:
                already = already or neighbor == hit
            if not already:
                hits.append(neighbor)
        if current_count == len(hits):
            return True
        for h in hits:
            g = self._make_group(objects, h, hits)
            if g:
                return True

    def make_group(self, objects):
        """
            Grouping of objects.
        """
        groups = []
        alredys = [False for l in objects]
        for k, object in enumerate(objects):
            if alredys[k]:
                continue
            hits = [k, ]
            self._make_group(objects, k, hits)
            group = []
            for hit in hits:
                alredys[hit] = True
                group.append(objects[hit])
            groups.append(group)
        return groups

    def skinning(self, context):
        # debug_time = datetime.datetime.today()
        if len(context.selected_objects) == 0:
            return {"CANCELLED"}
        objects = []  # Only Mesh
        for obj in context.selected_objects:
            if isinstance(obj.data, bpy.types.Mesh):
                objects.append(obj)
        if len(objects) == 0:
            return {"CANCELLED"}
        center = Vector((0, 0, 0))
        sample = 0
        top = -sys.float_info.max
        bottom = sys.float_info.max

        for obj in objects:
            b = self.bound_loc(obj)
            to = b[BOUND_TOP]
            bo = b[BOUND_BOTTOM]
            ce = b[BOUND_CENTER]
            center += ce
            sample += 1
            top = max(top, to)
            bottom = min(bottom, bo)
        center /= sample
        """
            Detect body
        """
        min_length = sys.float_info.max
        body = None
        for obj in objects:
            b = self.bound_loc(obj)
            ce = b[BOUND_CENTER]
            length = (ce - center).length_squared
            if length < min_length:
                min_length = length
                body = obj
        if body is None:
            return
        """
            Detect others
        """
        heads = []
        right_arms = []
        left_arms = []
        right_legs = []
        left_legs = []
        hips_height = self.lerp(bottom, top, 0.333)
        for obj in objects:
            body_bound = self.bound_loc(body)
            body_center = body_bound[BOUND_CENTER]
            # mat = Matrix(body.matrix_world)
            body_left = body_bound[BOUND_LEFT]
            body_right = body_bound[BOUND_RIGHT]
            body_radius = (body_left - body_right) * 0.5
            if body == obj:
                continue
            bound = self.bound_loc(obj)
            center = bound[BOUND_CENTER]
            radius = abs(center.x - body_center.x)
            if center.z < hips_height:
                if center.x < body_center.x:
                    right_legs.append(obj)
                else:
                    left_legs.append(obj)
            elif radius < body_radius:
                heads.append(obj)
            else:
                if center.x < body_center.x:
                    right_arms.append(obj)
                else:
                    left_arms.append(obj)
        """
            Create armature
        """
        bpy.ops.object.armature_add(
            location=(0.0, 0.0, 0.0), enter_editmode=True)
        # arma = context.active_object
        arma = context.view_layer.objects.active
        # context.object.show_x_ray = True
        context.object.show_in_front = True
        """
            Body bone
        """
        bone = arma.data.edit_bones[0]
        bone.name = "hips"
        hips, chest = self.create_bone(
            context, arma, bone, body, None,
            arma.data.edit_bones, bone_type=BONE_TYPE_BODY)
        """
            Leg bones
        """
        self.create_grouped_bone(
            context, right_legs, arma, hips, BONE_TYPE_LEG_RIGHT)
        self.create_grouped_bone(
            context, left_legs, arma, hips, BONE_TYPE_LEG_LEFT)

        head_groups = self.make_group(heads)
        for group in head_groups:
            primary_head = self.primary_obj(group)
            bone = arma.data.edit_bones.new("head")
            primary_bone = self.create_head(bone, primary_head, chest)
            for obj in group:
                if obj == primary_head:
                    continue
                bone = arma.data.edit_bones.new("head")
                self.create_bone(
                    context, arma, bone, obj, primary_bone,
                    arma.data.edit_bones, bone_type=BONE_TYPE_ANY)
        self.create_grouped_bone(
            context, right_arms,
            arma, chest, BONE_TYPE_ARM_RIGHT)
        self.create_grouped_bone(
            context, left_arms,
            arma, chest, BONE_TYPE_ARM_LEFT)
        if context.window_manager.love2d3d.armature_finger:
            """
                Rename fingers
            """
            left_fingers = []
            right_fingers = []
            for bone in arma.data.edit_bones:
                if bone.name.startswith("finger") and not bone.use_connect:
                    if bone.name.endswith(".L"):
                        left_fingers.append(bone)
                    else:
                        right_fingers.append(bone)
            lefts = sorted(left_fingers, key=lambda bone: bone.head.y)
            rights = sorted(right_fingers, key=lambda bone: bone.head.y)
            for k, bone4 in enumerate(lefts):
                bone4.name = "finger" + str(k) + ".04" + ".L"
                bone3 = bone4.children_recursive[0]
                bone3.name = "finger" + str(k) + ".03" + ".L"
                bone2 = bone3.children_recursive[0]
                bone2.name = "finger" + str(k) + ".02" + ".L"
                bone1 = bone2.children_recursive[0]
                bone1.name = "finger" + str(k) + ".01" + ".L"
            for k, bone4 in enumerate(rights):
                bone4.name = "finger" + str(k) + ".04" + ".R"
                bone3 = bone4.children_recursive[0]
                bone3.name = "finger" + str(k) + ".03" + ".R"
                bone2 = bone3.children_recursive[0]
                bone2.name = "finger" + str(k) + ".02" + ".R"
                bone1 = bone2.children_recursive[0]
                bone1.name = "finger" + str(k) + ".01" + ".R"
        for bone in arma.data.edit_bones:
            bone.select = True
        bpy.ops.armature.calculate_roll(type='GLOBAL_POS_Z')
        bpy.ops.object.mode_set(mode='OBJECT', toggle=False)
        for obj in objects:
            obj.select_set(True)
        # context.scene.objects.active = arma
        context.view_layer.objects.active = arma
        bpy.ops.object.parent_set(type='ARMATURE_AUTO')
        # print(datetime.datetime.today() - debug_time)
        return {'FINISHED'}

    def create_grouped_bone(self, context, objects,
                            armature, parent, bone_type):
        groups = self.make_group(objects)
        for group in groups:
            primary = self.primary_obj(group)
            bone = armature.data.edit_bones.new("bone")
            k, primary_bone = self.create_bone(
                context, armature, bone, primary,
                parent, armature.data.edit_bones, bone_type=bone_type)
            for obj in group:
                if obj == primary:
                    continue
                bone = armature.data.edit_bones.new("bone")
                self.create_bone(
                    context, armature, bone, obj,
                    primary_bone,
                    armature.data.edit_bones,
                    bone_type=BONE_TYPE_ANY)

    def lerp(self, start, end, ratio):
        return start * (1 - ratio) + end * ratio

    def invlerp(self, value, start, end, r):
        ratio = (value - start) / (end - start)
        i = int(ratio * r + 0.5)
        # i = min(max(0, i), r - 1)
        i = min(max(0, i), r)
        return i

    def debug_point(self, context, location, type='PLAIN_AXES'):
        o = context.blend_data.objects.new("P", None)
        o.location = location
        o.scale = (0.01, 0.01, 0.01)
        # context.scene.objects.link(o)
        context.scene.collection.objects.link(o)
        # o.empty_draw_type = type
        o.empty_display_type = type

    def create_bone(self, context, armature, bone, obj, chest,
                    bones, bone_type=BONE_TYPE_BODY, fingers=None):
        finger = fingers is not None
        if finger:
            bones.remove(bone)
        depsgraph = context.evaluated_depsgraph_get()
        object_eval = obj.evaluated_get(depsgraph)
        mesh_from_eval = object_eval.to_mesh()
        # mesh = obj.to_mesh()
        mesh = mesh_from_eval
        polygons = fingers if finger else mesh.polygons
        mat = Matrix(obj.matrix_world)
        if finger:
            polygons = fingers
            xs = [(mat @ Vector(polygon.center)).x for polygon in polygons]
            ys = [(mat @ Vector(polygon.center)).y for polygon in polygons]
            zs = [(mat @ Vector(polygon.center)).z for polygon in polygons]
            le = max(xs)
            ri = min(xs)
            ba = max(ys)
            fr = min(ys)
            to = max(zs)
            bo = min(zs)
        else:
            polygons = mesh.polygons
            b = self.bound_loc(obj)
            le = b[BOUND_LEFT]
            ri = b[BOUND_RIGHT]
            ba = b[BOUND_BACK]
            fr = b[BOUND_FRONT]
            to = b[BOUND_TOP]
            bo = b[BOUND_BOTTOM]
        ce = Vector((
            self.lerp(ri, le, 0.5), self.lerp(fr, ba, 0.5),
            self.lerp(bo, to, 0.5)))
        if bone_type == BONE_TYPE_BODY:
            # ce = b[BOUND_CENTER]
            body_top = Vector((ce.x, ce.y, to))
            body_bottom = Vector((ce.x, ce.y, bo))
        len_x = le - ri
        len_y = ba - fr
        len_z = to - bo
        love2d3d = context.window_manager.love2d3d
        armature_resolution = love2d3d.armature_resolution
        finger_resolution = love2d3d.armature_finger_resolution
        if finger:
            lattice = min(len_x, len_y, len_z) / finger_resolution  # units
        else:
            lattice = min(len_x, len_y, len_z) / armature_resolution  # units
        if lattice == 0.0:
            return None, None
        rx = int(len_x / lattice)  # x loop count
        ry = int(len_y / lattice)  # y loop count
        rz = int(len_z / lattice)  # z loop count
        rx = max(1, rx)
        ry = max(1, ry)
        rz = max(1, rz)
        mx = 1.0 / float(rx)
        my = 1.0 / float(ry)
        mz = 1.0 / float(rz)
        start = (0, 0, 0)
        if bone_type == BONE_TYPE_BODY:
            origin = body_bottom
        else:
            origin = Vector(chest.tail)
        centers = []
        """
            Volume separation process to reduce polygons' calculation.
        """
        loop = 2
        half_x = self.lerp(ri, le, 0.5)
        half_y = self.lerp(fr, ba, 0.5)
        half_z = self.lerp(bo, to, 0.5)
        p = loop
        cakes = [[[[] for x in range(p)] for y in range(p)]for z in range(p)]
        for polygon in polygons:  # Nearest polygon
            center = mat @ Vector(polygon.center)
            x = 0 if center.x <= half_x else 1
            y = 0 if center.y <= half_y else 1
            z = 0 if center.z <= half_z else 1
            cakes[z][y][x].append(polygon)
        kds = [[[None for x in range(loop)] for y in range(loop)]
               for z in range(loop)]
        for x in range(loop):
            for y in range(loop):
                for z in range(loop):
                    cake = cakes[z][y][x]
                    kd = KDTree(len(cake))
                    for i, polygon in enumerate(cake):
                        kd.insert(mat @ Vector(polygon.center), i)
                    kd.balance()
                    kds[z][y][x] = kd
        """
            Deciding process of inside points.
        """
        for x in range(rx + 1):
            for y in range(ry + 1):
                for z in range(rz + 1):
                    current = Vector((
                        self.lerp(ri, le, x * mx),
                        self.lerp(fr, ba, y * my),
                        self.lerp(bo, to, z * mz)))
                    min_polygon = None
                    s = 0 if current.x <= half_x else 1
                    t = 0 if current.y <= half_y else 1
                    u = 0 if current.z <= half_z else 1
                    co_find = (current.x, current.y, current.z)
                    co, index, dist = kds[u][t][s].find(co_find)
                    if index is None:
                        continue
                    min_polygon = cakes[u][t][s][index]
                    if min_polygon is None:
                        continue
                    normal = mat.to_quaternion() @ Vector(min_polygon.normal)
                    center = mat @ (min_polygon.center)
                    """
                        Approximation of polygon's region.
                    """
                    min_length = np.sqrt(min_polygon.area) * 0.5
                    vec = current - center
                    coeff = vec.dot(normal)  # Projection along Normal
                    vec -= coeff * normal
                    length = vec.length_squared  # This is mistake. But good.
                    close = coeff
                    if length < min_length and close < 0:
                        if finger:
                            loc = coeff * normal + center
                            lx, ly, lz = loc.xyz
                            u = self.invlerp(lx, ri, le, rx)
                            v = self.invlerp(ly, fr, ba, ry)
                            w = self.invlerp(lz, bo, to, rz)
                            centers.append((u, v, w))  # Volume thinning
                        else:
                            centers.append((x, y, z))
                            # lx, ly, lz = center.xyz
                            # u = self.invlerp(lx, ri, le, rx)
                            # v = self.invlerp(ly, fr, ba, ry)
                            # w = self.invlerp(lz, bo, to, rz)
                            # centers.append((u, v, w))
        """
            Getting process of the nearest point from origin.
        """
        center_kd = KDTree(len(centers))
        for i, c in enumerate(centers):
            x, y, z = c
            current = Vector((
                self.lerp(ri, le, x * mx),
                self.lerp(fr, ba, y * my),
                self.lerp(bo, to, z * mz)))
            center_kd.insert(current.xyz, i)
        center_kd.balance()
        co, index, dist = center_kd.find(origin.xyz)
        start = centers[index]
        x, y, z = start
        current = Vector((
            self.lerp(ri, le, x * mx),
            self.lerp(fr, ba, y * my),
            self.lerp(bo, to, z * mz)))
        if not finger:
            bone.head = current
        """
            Getting process of the farthest point.
        """
        end = start
        max_length = 0.0
        max_coeff = 0.0
        for c in centers:
            x, y, z = c
            current = Vector((
                self.lerp(ri, le, x * mx),
                self.lerp(fr, ba, y * my),
                self.lerp(bo, to, z * mz)))
            u, v, w = start
            s = Vector((
                self.lerp(ri, le, u * mx),
                self.lerp(fr, ba, v * my),
                self.lerp(bo, to, w * mz)))
            if finger:
                vec = chest.tail - chest.head
                coeff = (current - s).dot(vec)
                if max_coeff < coeff:
                    max_coeff = coeff
                    end = (x, y, z)
            else:
                length = (current - s).length_squared
                if max_length < length:
                    max_length = length
                    end = (x, y, z)
        sx, sy, sz = start

        def limit_hips(index):
            stem = (body_top - body_bottom)
            cx, cy, cz = centers[index]
            current = Vector((
                self.lerp(ri, le, cx * mx),
                self.lerp(fr, ba, cy * my),
                self.lerp(bo, to, cz * mz)))
            branch = current - body_bottom
            if stem.length_squared == 0.0 or branch.length_squared == 0.0:
                return False
            return stem.angle(branch) < self.hips_limit_angle

        if bone_type == BONE_TYPE_BODY:
            min_length = sys.float_info.max
            """
                Getting process of hips point.
            """
            co, index, dist = center_kd.find(origin.xyz, filter=limit_hips)
            if co is None:
                start_loc = body_bottom
            else:
                start_loc = Vector((
                    body_bottom.x,
                    body_bottom.y,
                    Vector(co).z))
        else:
            start_loc = Vector((
                self.lerp(ri, le, sx * mx),
                self.lerp(fr, ba, sy * my),
                self.lerp(bo, to, sz * mz)))
        if not finger:
            bone.head = start_loc
        ex, ey, ez = end
        if bone_type == BONE_TYPE_BODY:
            end_loc = body_top
        else:
            end_loc = Vector((
                self.lerp(ri, le, ex * mx),
                self.lerp(fr, ba, ey * my),
                self.lerp(bo, to, ez * mz)))
        """
            Getting process of center and neck point.
        """
        co, index, dist = center_kd.find(start_loc.lerp(end_loc, 0.5).xyz)
        center_loc = co
        co, index, dist = center_kd.find(start_loc.lerp(end_loc, 0.75).xyz)
        neck_loc = co
        if bone_type == BONE_TYPE_BODY:
            hips_loc = start_loc.lerp(end_loc, 0.25)
            bone.tail = hips_loc
            center_loc = start_loc.lerp(end_loc, 0.5)
            neck_loc = start_loc.lerp(end_loc, 0.75)
        elif finger:
            pass
        else:
            bone.tail = center_loc
            bone.parent = chest
        parent = bone
        start_bone = bone
        """
            Create process of primary bones.
        """
        if bone_type == BONE_TYPE_BODY:
            bone = bones.new("waist")
            bone.head = hips_loc
            bone.tail = center_loc
            bone.parent = parent
            bone.use_connect = True
            parent = bone
            bone = bones.new("chest")
            bone.head = center_loc
            bone.tail = neck_loc
            bone.parent = parent
            bone.use_connect = True
            parent = bone
            chest_bone = bone
            bone = bones.new("neck")
            bone.head = neck_loc
            bone.tail = end_loc
            bone.parent = parent
            bone.use_connect = True
            end_bone = bone

        elif bone_type == BONE_TYPE_FINGER_LEFT or\
                bone_type == BONE_TYPE_FINGER_RIGHT:
            name = "finger"
            end_bone = None
            start_bone = None
        else:
            if bone_type == BONE_TYPE_HEAD:
                name = "bone"
            elif bone_type == BONE_TYPE_ARM_LEFT:
                name = "shoulder.L"
            elif bone_type == BONE_TYPE_ARM_RIGHT:
                name = "shoulder.R"
            elif bone_type == BONE_TYPE_LEG_LEFT:
                name = "pelvis.L"
            elif bone_type == BONE_TYPE_LEG_RIGHT:
                name = "pelvis.R"
            else:
                name = "bone"
            bone.name = name
            vec = (origin - start_loc)
            basis = (start_loc - center_loc).normalized()
            coeff = vec.dot(basis)
            bone.head = 0.5 * coeff * basis + start_loc
            bone.tail = start_loc

            if bone_type == BONE_TYPE_HEAD:
                name = "bone"
            elif bone_type == BONE_TYPE_ARM_LEFT:
                name = "upper_arm.L"
            elif bone_type == BONE_TYPE_ARM_RIGHT:
                name = "upper_arm.R"
            elif bone_type == BONE_TYPE_LEG_LEFT:
                name = "thigh.L"
            elif bone_type == BONE_TYPE_LEG_RIGHT:
                name = "thigh.R"
            else:
                name = "bone"
            bone = bones.new(name)
            bone.head = start_loc
            bone.tail = center_loc
            bone.parent = parent
            bone.use_connect = True
            parent = bone
            if bone_type == BONE_TYPE_HEAD:
                name = "bone"
            elif bone_type == BONE_TYPE_ARM_LEFT:
                name = "forearm.L"
            elif bone_type == BONE_TYPE_ARM_RIGHT:
                name = "forearm.R"
            elif bone_type == BONE_TYPE_LEG_LEFT:
                name = "shin.L"
            elif bone_type == BONE_TYPE_LEG_RIGHT:
                name = "shin.R"
            else:
                name = "bone"
            bone = bones.new(name)
            bone.head = center_loc
            bone.tail = neck_loc
            bone.parent = parent
            bone.use_connect = True
            parent = bone
            if bone_type == BONE_TYPE_HEAD:
                name = "bone"
            elif bone_type == BONE_TYPE_ARM_LEFT:
                name = "hand.L"
            elif bone_type == BONE_TYPE_ARM_RIGHT:
                name = "hand.R"
            elif bone_type == BONE_TYPE_LEG_LEFT:
                name = "foot.L"
            elif bone_type == BONE_TYPE_LEG_RIGHT:
                name = "foot.R"
            else:
                name = "bone"
            bone = bones.new(name)
            bone.head = neck_loc
            bone.tail = end_loc
            bone.parent = parent
            bone.use_connect = True
            end_bone = bone

        """
            Caluculate process of volume.
        """
        volume_xs = [0 for x in range(rx + 1)]
        volume_ys = [0 for y in range(ry + 1)]
        volume_zs = [0 for z in range(rz + 1)]
        unit_x = my * mz
        unit_y = mz * mx
        unit_z = mx * my
        for center in centers:
            x, y, z = center
            volume_xs[x] += unit_x
            volume_ys[y] += unit_y
            volume_zs[z] += unit_z
        hit_xs = []
        hit_ys = []
        hit_zs = []
        threshopld = 1.0
        ratio = self.finger_branch_boost if finger else self.branch_boost
        soft_x = unit_x * 0.0001
        soft_y = unit_y * 0.0001
        soft_z = unit_z * 0.0001
        """
            Differential process of volume in log scale.
            It tell us like "A's scale is B's scale of x1, x10, x100...".
        """
        for x in range(1, rx + 1 - 1):
            vm = volume_xs[x - 1]
            v0 = volume_xs[x]
            v1 = volume_xs[x + 1]
            sv = np.log2(v0 + soft_x) * ratio
            ev = np.log2(v1 + soft_x) * ratio
            mv = np.log2(vm + soft_x) * ratio
            diff = (2 * sv - ev - mv) ** 2
            if threshopld <= diff:
                hit_xs.append(x)
        for y in range(1, ry + 1 - 1):
            vm = volume_ys[y - 1]
            v0 = volume_ys[y]
            v1 = volume_ys[y + 1]
            sv = np.log2(v0 + soft_y) * ratio
            ev = np.log2(v1 + soft_y) * ratio
            mv = np.log2(vm + soft_y) * ratio
            diff = (2 * sv - ev - mv) ** 2
            if threshopld <= diff:
                hit_ys.append(y)
        for z in range(1, rz + 1 - 1):
            vm = volume_zs[z - 1]
            v0 = volume_zs[z]
            v1 = volume_zs[z + 1]
            sv = np.log2(v0 + soft_z) * ratio
            ev = np.log2(v1 + soft_z) * ratio
            mv = np.log2(vm + soft_z) * ratio
            diff = (2 * sv - ev - mv) ** 2
            if threshopld <= diff:
                hit_zs.append(z)
        """
            Diciding process of branch point like fingers.
        """
        hits = []
        average = Vector((0, 0, 0))
        dispersion = 0.0
        sum = 0
        center_limit = self.center_limit_angle
        for x in hit_xs:
            for y in hit_ys:
                for z in hit_zs:
                    for center in centers:
                        u, v, w = center
                        current = Vector((
                            self.lerp(ri, le, u * mx),
                            self.lerp(fr, ba, v * my),
                            self.lerp(bo, to, w * mz)))
                        free = True
                        """
                            Avoiding process of body's neck.
                            It is because bad points for shoulders.
                        """
                        if bone_type == BONE_TYPE_BODY:
                            proj0 = Vector((
                                current.x,
                                body_bottom.y,
                                current.z))
                            proj1 = Vector((current.x, body_top.y, current.z))
                            branch0 = (proj0 - body_bottom)
                            branch1 = (proj1 - body_top)
                            stem = (body_top - body_bottom)
                            if branch0.length_squared == 0.0 or\
                               branch1.length_squared == 0.0:
                                continue
                            angle0 = stem.angle(branch0)
                            angle1 = stem.angle(branch1)
                            free = center_limit < angle0 and\
                                center_limit < angle1
                        if x == u and y == v and z == w and free:
                            hits.append((u, v, w))
                            average += current
                            dispersion += current.length_squared
                            sum += 1
        """
            Gathering process of branch points.
        """
        if sum == 0:
            return start_bone, end_bone
        average /= sum
        dispersion /= sum
        dispersion -= average.length_squared
        finger_ratio = (dispersion * self.finger_gather_ratio * 0.01)
        body_ratio = (dispersion * self.gather_ratio * 0.01)
        gather_ratio = finger_ratio if finger else body_ratio
        bound = le, ri, ba, fr, to, bo
        m = mx, my, mz
        gathers = self.gather_point(hits, bound, m, gather_ratio)
        joints = []
        """
            Averaging process of gatherd branch points.
        """
        for gather in gathers:
            average = Vector((0, 0, 0))
            sum = 0
            for point in gather:
                px, py, pz = hits[point]
                p_loc = Vector((
                    self.lerp(ri, le, px * mx),
                    self.lerp(fr, ba, py * my),
                    self.lerp(bo, to, pz * mz)))
                average += p_loc
                sum += 1
            if sum == 0:
                continue
            average /= sum
            joints.append(average)
        """
            Calculating and creating process of bones.
        """
        def create_joint(centers, hinges, bounds, rs, ms, dispersion, name,
                         parent, bone_type, end=Vector((0, 0, 0))):
            le, ri, ba, fr, to, bo = bounds
            rx, ry, rz = rs
            ms = (mx, my, mz)
            tips = [(
                self.invlerp(hinge[1].x, ri, le, rx),
                self.invlerp(hinge[1].y, fr, ba, ry),
                self.invlerp(hinge[1].z, bo, to, rz)) for hinge in hinges]
            if bone_type == BONE_TYPE_FINGER_LEFT or\
               bone_type == BONE_TYPE_FINGER_RIGHT:
                gathers = self.gather_point(tips, bounds, ms,
                                            dispersion, parent=parent, end=end)
            else:
                gathers = self.gather_point(tips, bounds, ms, dispersion)
            """
                Averaging process of gathered tips.
            """
            averages = [Vector((0, 0, 0)) for g in gathers]
            for k, gather in enumerate(gathers):
                if bone_type == BONE_TYPE_FINGER_LEFT or\
                   bone_type == BONE_TYPE_FINGER_RIGHT:
                    average = Vector((0, 0, 0))
                    max_close = 0.0
                    max_point = Vector((0, 0, 0))
                    vec = end - parent.head
                    for point in gather:
                        px, py, pz = tips[point]
                        p_loc = Vector((
                            self.lerp(ri, le, px * mx),
                            self.lerp(fr, ba, py * my),
                            self.lerp(bo, to, pz * mz)))
                        close = p_loc.dot(vec)
                        if max_close < close:
                            max_close = close
                            max_point = p_loc
                    averages[k] = max_point
                else:
                    average = Vector((0, 0, 0))
                    sum = 0
                    for point in gather:
                        px, py, pz = tips[point]
                        p_loc = Vector((
                            self.lerp(ri, le, px * mx),
                            self.lerp(fr, ba, py * my),
                            self.lerp(bo, to, pz * mz)))
                        average += p_loc
                        sum += 1
                    average /= sum
                    averages[k] = average
            """
                Grouping hinge by average points.
            """
            groups = [[] for a in averages]
            for hinge in hinges:
                joint, tip = hinge
                min_length = sys.float_info.max
                min_index = 0
                for k, average in enumerate(averages):
                    length = (average - tip).length_squared
                    if length < min_length:
                        min_length = length
                        min_index = k
                groups[min_index].append(joint)
            end_bones = []
            for k, tip in enumerate(averages):
                max_length = -sys.float_info.max
                max_joint = tip
                group = groups[k]
                if len(group) == 0:
                    continue
                for joint in group:
                    length = (joint - tip).length_squared
                    if max_length < length:
                        max_length = length
                        max_joint = joint
                """
                    Elbow
                """
                f = max_joint.lerp(tip, 0.5).xyz
                co, index, dist = center_kd.find(f)
                min_e = co
                """
                    Hand neck
                """
                f = max_joint.lerp(tip, 0.75).xyz
                co, index, dist = center_kd.find(f)
                min_n = co
                """
                    Finger tip
                """
                f = max_joint.lerp(tip, 0.875).xyz
                co, index, dist = center_kd.find(f)
                min_t = co
                """
                    Shoulder
                """
                vec = (max_joint - min_e).normalized()
                coeff = (parent.tail - max_joint).dot(vec)
                s = vec * coeff * 0.5 + max_joint
                co, index, dist = center_kd.find(s.xyz)
                min_s = co
                if bone_type == BONE_TYPE_HEAD:
                    name = "bone"
                elif bone_type == BONE_TYPE_ARM_LEFT:
                    name = "shoulder.L"
                elif bone_type == BONE_TYPE_ARM_RIGHT:
                    name = "shoulder.R"
                elif bone_type == BONE_TYPE_LEG_LEFT:
                    name = "pelvis.L"
                elif bone_type == BONE_TYPE_LEG_RIGHT:
                    name = "pelvis.R"
                elif bone_type == BONE_TYPE_FINGER_LEFT:
                    name = "finger" + self.index_bone(max_joint) + ".L"
                elif bone_type == BONE_TYPE_FINGER_RIGHT:
                    name = "finger" + self.index_bone(max_joint) + ".R"
                else:
                    name = "bone"

                if bone_type != BONE_TYPE_FINGER_LEFT and\
                   bone_type != BONE_TYPE_FINGER_RIGHT:
                    bone = bones.new(name)
                    bone.head = min_s
                    bone.tail = max_joint
                    bone.parent = parent
                    p = bone
                if bone_type == BONE_TYPE_HEAD:
                    name = "bone"
                elif bone_type == BONE_TYPE_ARM_LEFT:
                    name = "upper_arm.L"
                elif bone_type == BONE_TYPE_ARM_RIGHT:
                    name = "upper_arm.R"
                elif bone_type == BONE_TYPE_LEG_LEFT:
                    name = "thigh.L"
                elif bone_type == BONE_TYPE_LEG_RIGHT:
                    name = "thigh.R"
                elif bone_type == BONE_TYPE_FINGER_LEFT:
                    name = "finger" + self.index_bone(max_joint) + ".L"
                elif bone_type == BONE_TYPE_FINGER_RIGHT:
                    name = "finger" + self.index_bone(max_joint) + ".R"
                else:
                    name = "bone"
                bone = bones.new(name)
                bone.head = max_joint
                bone.tail = min_e
                if bone_type == BONE_TYPE_FINGER_LEFT or\
                   bone_type == BONE_TYPE_FINGER_RIGHT:
                    bone.parent = parent
                else:
                    bone.parent = p
                    bone.use_connect = True
                p = bone
                if bone_type == BONE_TYPE_HEAD:
                    name = "bone"
                elif bone_type == BONE_TYPE_ARM_LEFT:
                    name = "forearm.L"
                elif bone_type == BONE_TYPE_ARM_RIGHT:
                    name = "forearm.R"
                elif bone_type == BONE_TYPE_LEG_LEFT:
                    name = "shin.L"
                elif bone_type == BONE_TYPE_LEG_RIGHT:
                    name = "shin.R"
                elif bone_type == BONE_TYPE_FINGER_LEFT:
                    name = "finger" + self.index_bone(max_joint) + ".L"
                elif bone_type == BONE_TYPE_FINGER_RIGHT:
                    name = "finger" + self.index_bone(max_joint) + ".R"
                else:
                    name = "bone"
                bone = bones.new(name)
                bone.head = min_e
                bone.tail = min_n
                bone.parent = p
                bone.use_connect = True
                p = bone
                if bone_type == BONE_TYPE_HEAD:
                    name = "bone"
                elif bone_type == BONE_TYPE_ARM_LEFT:
                    name = "hand.L"
                elif bone_type == BONE_TYPE_ARM_RIGHT:
                    name = "hand.R"
                elif bone_type == BONE_TYPE_LEG_LEFT:
                    name = "foot.L"
                elif bone_type == BONE_TYPE_LEG_RIGHT:
                    name = "foot.R"
                elif bone_type == BONE_TYPE_FINGER_LEFT:
                    name = "finger" + self.index_bone(max_joint) + ".L"
                elif bone_type == BONE_TYPE_FINGER_RIGHT:
                    name = "finger" + self.index_bone(max_joint) + ".R"
                else:
                    name = "bone"
                bone = bones.new(name)
                bone.head = min_n
                if bone_type == BONE_TYPE_FINGER_LEFT or\
                   bone_type == BONE_TYPE_FINGER_RIGHT:
                    bone.tail = min_t
                else:
                    bone.tail = tip
                bone.parent = p
                bone.use_connect = True
                p = bone
                if bone_type == BONE_TYPE_FINGER_LEFT:
                    name = "finger" + self.index_bone(max_joint) + ".L"
                    bone = bones.new(name)
                    bone.head = min_n
                    bone.tail = tip
                    bone.parent = p
                    bone.use_connect = True
                elif bone_type == BONE_TYPE_FINGER_RIGHT:
                    name = "finger" + self.index_bone(max_joint) + ".R"
                    bone = bones.new(name)
                    bone.head = min_n
                    bone.tail = tip
                    bone.parent = p
                    bone.use_connect = True
                end_bones.append(bone)

            return end_bones
        """
            Getting process of tips like fingers's tip.
        """
        left_hands = []
        right_hands = []
        left_foots = []
        right_foots = []
        if bone_type == BONE_TYPE_BODY:
            right_arms = []
            left_arms = []
            right_legs = []
            left_legs = []
            for joint in joints:
                arm = hips_loc.z < joint.z
                left = 0 < joint.x - center_loc.x
                if arm:
                    stem = neck_loc - center_loc
                    branch = joint - center_loc
                else:
                    stem = start_loc - center_loc
                    branch = joint - center_loc
                if stem.length_squared == 0.0 or branch.length_squared == 0.0:
                    continue
                branch_normal = branch.normalized()
                angle = stem.angle(branch)
                arm_angle = self.arm_limit_angle
                leg_angle = self.leg_limit_angle
                limit_angle = arm_angle if arm else leg_angle
                if limit_angle < angle:
                    max_close = -sys.float_info.max
                    max_loc = joint
                    for center in centers:
                        x, y, z = center
                        current = Vector((
                            self.lerp(ri, le, x * mx),
                            self.lerp(fr, ba, y * my),
                            self.lerp(bo, to, z * mz)))
                        if arm:
                            height = hips_loc.z < current.z
                        else:
                            height = current.z < hips_loc.z
                        close = (current - joint).dot(branch_normal)
                        if height and max_close < close:
                            max_close = close
                            max_loc = current
                    if arm:
                        if left:
                            left_arms.append((joint, max_loc))
                        else:
                            right_arms.append((joint, max_loc))
                    else:
                        if left:
                            left_legs.append((joint, max_loc))
                        else:
                            right_legs.append((joint, max_loc))
            bounds = (le, ri, ba, fr, to, bo)
            ms = (mx, my, mz)
            rs = (rx, ry, rz)
            left_hands = create_joint(
                centers, left_arms, bounds, rs, ms,
                gather_ratio, "Arm.L", chest_bone, BONE_TYPE_ARM_LEFT)
            right_hands = create_joint(
                centers, right_arms, bounds, rs, ms,
                gather_ratio, "Arm.R", chest_bone, BONE_TYPE_ARM_RIGHT)
            left_foots = create_joint(
                centers, left_legs, bounds, rs, ms,
                gather_ratio, "Leg.L", start_bone, BONE_TYPE_LEG_LEFT)
            right_foots = create_joint(
                centers, right_legs, bounds, rs, ms,
                gather_ratio, "Leg.R", start_bone, BONE_TYPE_LEG_RIGHT)
        elif (bone_type == BONE_TYPE_FINGER_LEFT or
              bone_type == BONE_TYPE_FINGER_RIGHT):
            tips = []
            for joint in joints:
                stem = end_loc - chest.head
                branch = joint - chest.head
                if stem.length_squared == 0.0 or branch.length_squared == 0.0:
                    continue
                branch_normal = branch.normalized()
                angle = stem.angle(branch)
                limit_angle = self.finger_limit_angle
                close = stem.dot(branch)
                if 0 < close:
                    max_close = -sys.float_info.max
                    max_loc = joint
                    for center in centers:
                        x, y, z = center
                        current = Vector((
                            self.lerp(ri, le, x * mx),
                            self.lerp(fr, ba, y * my),
                            self.lerp(bo, to, z * mz)))
                        vec0 = current - joint
                        close = vec0.dot(stem)
                        if vec0.length_squared == 0.0:
                            continue
                        angle = vec0.angle(stem)
                        if max_close < close and angle < limit_angle:
                            max_close = close
                            max_loc = current
                    tips.append((joint, max_loc))
            bounds = (le, ri, ba, fr, to, bo)
            ms = (mx, my, mz)
            rs = (rx, ry, rz)
            tip_gather_ratio = dispersion * self.tip_gather_ratio * 0.01
            create_joint(centers, tips, bounds, rs, ms,
                         tip_gather_ratio, name, chest, bone_type, end=end_loc)
        else:
            tips = []
            for joint in joints:
                stem = end_loc - neck_loc
                branch = joint - neck_loc
                if stem.length_squared == 0.0 or branch.length_squared == 0.0:
                    continue
                branch_normal = branch.normalized()
                angle = stem.angle(branch)
                limit_angle = self.any_limit_angle
                close = stem.dot(branch)
                if 0 < close and limit_angle < angle:
                    max_close = -sys.float_info.max
                    max_loc = joint
                    for center in centers:
                        x, y, z = center
                        current = Vector((
                            self.lerp(ri, le, x * mx),
                            self.lerp(fr, ba, y * my),
                            self.lerp(bo, to, z * mz)))
                        close = (current - joint).dot(branch_normal)
                        if max_close < close:
                            max_close = close
                            max_loc = current
                    tips.append((joint, max_loc))
            bounds = (le, ri, ba, fr, to, bo)
            ms = (mx, my, mz)
            rs = (rx, ry, rz)
            if (bone_type == BONE_TYPE_ARM_LEFT or
                bone_type == BONE_TYPE_ARM_RIGHT) and\
               context.window_manager.love2d3d.armature_finger:
                pass
            else:
                create_joint(centers, tips, bounds, rs, ms, gather_ratio,
                             name, end_bone, BONE_TYPE_ANY)
            if bone_type == BONE_TYPE_ARM_LEFT:
                left_hands.append(end_bone)
            elif bone_type == BONE_TYPE_ARM_RIGHT:
                right_hands.append(end_bone)
        """
            Finger process.
        """
        if not finger and context.window_manager.love2d3d.armature_finger:
            self.create_finger(context, armature, obj, polygons, mat, bones,
                               left_hands, BONE_TYPE_FINGER_LEFT)
            self.create_finger(context, armature, obj, polygons, mat, bones,
                               right_hands, BONE_TYPE_FINGER_RIGHT)
        """
            Finish process.
        """
        if not finger:
            object_eval.to_mesh_clear()
        return start_bone, end_bone

    def index_bone(self, location):
        y = int(location.y * 1000)
        return str(y)

    def create_finger(self, context, armature, obj, polygons,
                      mat, bones, hands, bone_type):
        for hand in hands:
            fingers = []
            for polygon in polygons:
                center = mat @ Vector(polygon.center)
                vec = hand.tail - hand.head
                m = hand.head.lerp(hand.tail, 0.0)
                coeff = (center - m).dot(vec)
                vec1 = center - hand.head
                if vec.length_squared == 0.0 or vec1.length_squared == 0.0:
                    continue
                angle = vec.angle(vec1)
                if 0 < coeff and angle < self.hand_limit_angle:
                    fingers.append(polygon)
            if len(fingers) != 0:
                bone = armature.data.edit_bones.new("head")
                self.create_bone(context, armature, bone, obj, hand, bones,
                                 bone_type=bone_type, fingers=fingers)

    def gather_point(self, points, bound, m, dispersion,
                     parent=None, end=Vector((0, 0, 0))):
        """
            Gathering points to gathers.
        """
        gathers = []
        alreadys = [False for p in points]
        for k, point in enumerate(points):
            if alreadys[k]:
                continue
            hits = [k, ]
            self._gather_point(k, points, bound, m, dispersion,
                               hits, parent=parent, end=end)
            gathers.append(hits)
            for hit in hits:
                alreadys[hit] = True
        return gathers

    def _gather_point(self, index, points, bound, m,
                      dispersion, hits, parent=None, end=Vector((0, 0, 0))):
        """
            Recursion call of points' collision.
        """
        point = points[index]
        current_count = len(hits)
        px, py, pz = point
        le, ri, ba, fr, to, bo = bound
        mx, my, mz = m
        if parent is not None:
            vec = (end - parent.head).normalized()
            p_loc = Vector((
                self.lerp(ri, le, px * mx),
                self.lerp(fr, ba, py * my),
                self.lerp(bo, to, pz * mz)))
            coeff = (p_loc - parent.head).dot(vec)
            p_loc = (p_loc - parent.head) - coeff * vec
        else:
            p_loc = Vector((
                self.lerp(ri, le, px * mx),
                self.lerp(fr, ba, py * my),
                self.lerp(bo, to, pz * mz)))
        neighbors = []
        for k, neighbor in enumerate(points):
            if neighbor == point:
                continue
            nx, ny, nz = neighbor
            if parent is not None:
                vec = (end - parent.head).normalized()
                n_loc = Vector((
                    self.lerp(ri, le, nx * mx),
                    self.lerp(fr, ba, ny * my),
                    self.lerp(bo, to, nz * mz)))
                coeff = (n_loc - parent.head).dot(vec)
                n_loc = (n_loc - parent.head) - coeff * vec
            else:
                n_loc = Vector((
                    self.lerp(ri, le, nx * mx),
                    self.lerp(fr, ba, ny * my),
                    self.lerp(bo, to, nz * mz)))
            length = (p_loc - n_loc).length_squared
            if length < dispersion:
                neighbors.append(k)
        for neighbor in neighbors:
            already = False
            for hit in hits:
                already = already or hit == neighbor
            if not already:
                hits.append(neighbor)
        if current_count == len(hits):
            return True
        for neighbor in neighbors:
            g = self._gather_point(neighbor, points, bound, m,
                                   dispersion, hits, parent=parent, end=end)
            if g:
                return True

    def create_head(self, bone, obj, chest):
        b = self.bound_loc(obj)
        p = b[BOUND_TOP]
        n = b[BOUND_BOTTOM]
        bone.head = Vector((obj.location.x, obj.location.y, n))
        bone.tail = Vector((obj.location.x, obj.location.y, p))
        bone.parent = chest
        return bone


class LOVE2D3D_MT_menu(bpy.types.Menu):
    bl_idname = "LOVE2D3D_MT_menu"
    bl_label = "Love2D3D"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Tool"
    bl_context = "objectmode"

    def draw(self, context):
        layout = self.layout
        layout.separator()
        layout.operator("image.open", icon="FILE_IMAGE", text="Open")
        layout.separator()
        layout.operator(LOVE2D3D_OT_createObject.bl_idname,
                        text="Create", icon="OUTLINER_OB_MESH")
        layout.separator()
        layout.operator(LOVE2D3D_OT_createArmature.bl_idname,
                        text="Create", icon="OUTLINER_OB_ARMATURE")


class LOVE2D3D_PT_panel(bpy.types.Panel):
    bl_idname = "LOVE2D3D_PT_panel"
    bl_label = "Love2D3D"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "View"
    bl_context = "objectmode"

    def draw(self, context):
        layout = self.layout
        preview = context.window_manager.love2d3d.preview
        icon = 'HIDE_OFF' if preview else 'HIDE_ON'
        layout.operator(LOVE2D3D_OT_preview.bl_idname,
                        text="Preview", icon=icon)
        row = layout.row()
        row.prop_search(context.window_manager.love2d3d,
                        "image_front", context.blend_data, "images")
        row.operator("image.open", icon="FILEBROWSER", text="")
        row.operator("image.new", icon="DUPLICATE", text="")
        layout.prop(context.window_manager.love2d3d, "view_align")


class Love2D3DProps(bpy.types.PropertyGroup):
    image_front: StringProperty(name="Front",
                                description="Front image of mesh")
    image_back: StringProperty(name="Back",
                               description="Back image of mesh")
    rough: IntProperty(name="Rough",
                       description="Roughness of image", min=1,
                       default=8, subtype="PIXEL")
    smooth: IntProperty(name="Smooth",
                        description="Smoothness of mesh",
                        min=1, default=30)
    scale: FloatProperty(name="Scale",
                         description="Length per pixel",
                         unit="LENGTH", min=0.001,
                         default=0.01, precision=4)
    depth_front: FloatProperty(name="Front",
                               description="Depth of front face",
                               unit="NONE", min=0, default=1)
    depth_back: FloatProperty(name="Back",
                              description="Depth of back face",
                              unit="NONE", min=0, default=1)
    fat: FloatProperty(name="Fat",
                       description="Fat of mesh",
                       default=0.2, min=0.0)
    modifier: BoolProperty(name="Modifier",
                           description="Apply modifiers to object",
                           default=True)
    threshold: FloatProperty(name="Threshold",
                             description="Threshold of background" +
                                         "in image",
                             min=0.0, max=1.0,
                             default=0.0, subtype="FACTOR")
    opacity: BoolProperty(name="Opacity",
                          description="Use Opacity for threshold")
    view_align: BoolProperty(name="View align",
                             description="Use view align for mesh")
    preview: BoolProperty(name="Preview",
                          description="Use preview for mesh now",
                          options={'HIDDEN'})
    decimate: BoolProperty(name="Decimate",
                           description="Use decimate modifier to object",
                           default=False)
    decimate_ratio: FloatProperty(name="Ratio",
                                  description="Decimate ratio",
                                  default=0.2, min=0.0,
                                  max=1.0, subtype="FACTOR")
    shadeless: BoolProperty(name="Shadeless",
                            description="Use shadeless in" +
                                        " object's material",
                            default=True)
    armature_resolution: FloatProperty(name="Resolution",
                                       description="Resolution of" +
                                                   "calculation",
                                       min=1, default=6.0)
    armature_finger_resolution: FloatProperty(name="Finger resolution",
                                              description="Finger's" +
                                                          "resolution" +
                                                          "of calculation",
                                              min=1, default=6.0)
    armature_finger: BoolProperty(name="Finger",
                                  description="Use finger in armature",
                                  default=False)


classes = [
    LOVE2D3D_OT_preview,
    LOVE2D3D_OT_createObject,
    LOVE2D3D_OT_createArmature,
    LOVE2D3D_PT_panel,
    LOVE2D3D_MT_menu,
    Love2D3DProps
]


def draw_mesh_item(self, context):
    layout = self.layout
    layout.operator(LOVE2D3D_OT_createObject.bl_idname,
                    text="Love2D3D", icon="MESH_UVSPHERE")


def draw_armature_item(self, context):
    layout = self.layout
    layout.operator(LOVE2D3D_OT_createArmature.bl_idname,
                    text="Love2D3D", icon="OUTLINER_OB_ARMATURE")


def draw_item(self, context):
    layout = self.layout
    layout.separator()
    layout.menu(LOVE2D3D_MT_menu.bl_idname, icon='PLUGIN')


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.VIEW3D_MT_mesh_add.append(draw_mesh_item)
    bpy.types.VIEW3D_MT_armature_add.append(draw_armature_item)
    bpy.types.WindowManager.love2d3d \
        = bpy.props.PointerProperty(type=Love2D3DProps)


def unregister():
    bpy.types.VIEW3D_MT_mesh_add.remove(draw_mesh_item)
    bpy.types.VIEW3D_MT_armature_add.remove(draw_armature_item)
    del bpy.types.WindowManager.love2d3d
    for c in classes:
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
