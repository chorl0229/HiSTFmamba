import torch


def whole2parts(motion, mode="t2m", window_size=None):
    device = motion.device

    if mode == "t2m":
        batch_size = motion.shape[0]
        nframes = motion.shape[1]
        aug_data = motion  # (batch_size, nframes, 263)
        joints_num = 22
        s = 0  # start
        e = 4  # end
        root_data = aug_data[..., s:e]  # (batch_size, nframes, 4)
        s = e
        e = e + (joints_num - 1) * 3
        ric_data = aug_data[..., s:e]  # (batch_size, nframes, (joints_num - 1) * 3)
        s = e
        e = e + (joints_num - 1) * 6
        rot_data = aug_data[..., s:e]  # (batch_size, nframes, (joints_num - 1) * 6)
        s = e
        e = e + joints_num * 3
        local_vel = aug_data[..., s:e]  # (batch_size, nframes, joints_num * 3)
        s = e
        e = e + 4
        feet = aug_data[..., s:e]  # (batch_size, nframes, 4)

        R_L_idx = torch.tensor([2, 5, 8, 11], dtype=torch.int64, device=device)
        L_L_idx = torch.tensor([1, 4, 7, 10], dtype=torch.int64, device=device)
        B_idx = torch.tensor([3, 6, 9, 12, 15], dtype=torch.int64, device=device)
        R_A_idx = torch.tensor([9, 14, 17, 19, 21], dtype=torch.int64, device=device)
        L_A_idx = torch.tensor([9, 13, 16, 18, 20], dtype=torch.int64, device=device)

        if window_size is not None:
            assert nframes == window_size, "nframes must equal window_size"

        # Reshape
        ric_data = ric_data.view(
            batch_size, nframes, joints_num - 1, 3
        )  # (batch_size, nframes, 21, 3)
        rot_data = rot_data.view(
            batch_size, nframes, joints_num - 1, 6
        )  # (batch_size, nframes, 21, 6)
        local_vel = local_vel.view(
            batch_size, nframes, joints_num, 3
        )  # (batch_size, nframes, 22, 3)

        # Concatenate root_data and the velocity of the first joint
        root_data = torch.cat(
            [root_data, local_vel[:, :, 0, :]], dim=-1
        )  # (batch_size, nframes, 7)

        # Adjust indices
        R_L_idx -= 1
        L_L_idx -= 1
        B_idx -= 1
        R_A_idx -= 1
        L_A_idx -= 1

        # Get parts data
        R_L = torch.cat(
            [
                ric_data[:, :, R_L_idx, :],  # (batch_size, nframes, num_joints, 3)
                rot_data[:, :, R_L_idx, :],  # (batch_size, nframes, num_joints, 6)
                local_vel[:, :, R_L_idx + 1, :],  # (batch_size, nframes, num_joints, 3)
            ],
            dim=-1,
        )  # (batch_size, nframes, num_joints, 12)

        L_L = torch.cat(
            [
                ric_data[:, :, L_L_idx, :],
                rot_data[:, :, L_L_idx, :],
                local_vel[:, :, L_L_idx + 1, :],
            ],
            dim=-1,
        )

        B = torch.cat(
            [
                ric_data[:, :, B_idx, :],
                rot_data[:, :, B_idx, :],
                local_vel[:, :, B_idx + 1, :],
            ],
            dim=-1,
        )

        R_A = torch.cat(
            [
                ric_data[:, :, R_A_idx, :],
                rot_data[:, :, R_A_idx, :],
                local_vel[:, :, R_A_idx + 1, :],
            ],
            dim=-1,
        )

        L_A = torch.cat(
            [
                ric_data[:, :, L_A_idx, :],
                rot_data[:, :, L_A_idx, :],
                local_vel[:, :, L_A_idx + 1, :],
            ],
            dim=-1,
        )

        # Reshape and concatenate feet information
        Root = root_data
        R_Leg = torch.cat(
            [R_L.view(batch_size, nframes, -1), feet[:, :, 2:]], dim=-1
        )  # (batch_size, nframes, 50)
        L_Leg = torch.cat(
            [L_L.view(batch_size, nframes, -1), feet[:, :, :2]], dim=-1
        )  # (batch_size, nframes, 50)
        Backbone = B.view(batch_size, nframes, -1)  # (batch_size, nframes, 60)
        R_Arm = R_A.view(batch_size, nframes, -1)  # (batch_size, nframes, 60)
        L_Arm = L_A.view(batch_size, nframes, -1)  # (batch_size, nframes, 60)

        return [Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm]

    elif mode == "kit":
        batch_size = motion.shape[0]
        nframes = motion.shape[1]
        aug_data = motion  # (batch_size, nframes, 251)
        joints_num = 21

        s = 0  # start
        e = 4  # end
        root_data = aug_data[..., s:e]  # (batch_size, nframes, 4)
        s = e
        e = e + (joints_num - 1) * 3
        ric_data = aug_data[..., s:e]  # (batch_size, nframes, (joints_num - 1) * 3)
        s = e
        e = e + (joints_num - 1) * 6
        rot_data = aug_data[..., s:e]  # (batch_size, nframes, (joints_num - 1) * 6)
        s = e
        e = e + joints_num * 3
        local_vel = aug_data[..., s:e]  # (batch_size, nframes, joints_num * 3)
        s = e
        e = e + 4
        feet = aug_data[..., s:e]  # (batch_size, nframes, 4)

        R_L_idx = torch.tensor([11, 12, 13, 14, 15], dtype=torch.int64, device=device)
        L_L_idx = torch.tensor([16, 17, 18, 19, 20], dtype=torch.int64, device=device)
        B_idx = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device=device)
        R_A_idx = torch.tensor([3, 5, 6, 7], dtype=torch.int64, device=device)
        L_A_idx = torch.tensor([3, 8, 9, 10], dtype=torch.int64, device=device)

        if window_size is not None:
            assert nframes == window_size, "nframes must equal window_size"

        # Reshape
        ric_data = ric_data.view(
            batch_size, nframes, joints_num - 1, 3
        )  # (batch_size, nframes, 20, 3)
        rot_data = rot_data.view(
            batch_size, nframes, joints_num - 1, 6
        )  # (batch_size, nframes, 20, 6)
        local_vel = local_vel.view(
            batch_size, nframes, joints_num, 3
        )  # (batch_size, nframes, 21, 3)

        # Concatenate root_data and the velocity of the first joint
        root_data = torch.cat(
            [root_data, local_vel[:, :, 0, :]], dim=-1
        )  # (batch_size, nframes, 7)

        # Adjust indices
        R_L_idx -= 1
        L_L_idx -= 1
        B_idx -= 1
        R_A_idx -= 1
        L_A_idx -= 1

        # Get parts data
        R_L = torch.cat(
            [
                ric_data[:, :, R_L_idx, :],  # (batch_size, nframes, num_joints, 3)
                rot_data[:, :, R_L_idx, :],  # (batch_size, nframes, num_joints, 6)
                local_vel[:, :, R_L_idx + 1, :],  # (batch_size, nframes, num_joints, 3)
            ],
            dim=-1,
        )  # (batch_size, nframes, num_joints, 12)

        L_L = torch.cat(
            [
                ric_data[:, :, L_L_idx, :],
                rot_data[:, :, L_L_idx, :],
                local_vel[:, :, L_L_idx + 1, :],
            ],
            dim=-1,
        )

        B = torch.cat(
            [
                ric_data[:, :, B_idx, :],
                rot_data[:, :, B_idx, :],
                local_vel[:, :, B_idx + 1, :],
            ],
            dim=-1,
        )

        R_A = torch.cat(
            [
                ric_data[:, :, R_A_idx, :],
                rot_data[:, :, R_A_idx, :],
                local_vel[:, :, R_A_idx + 1, :],
            ],
            dim=-1,
        )

        L_A = torch.cat(
            [
                ric_data[:, :, L_A_idx, :],
                rot_data[:, :, L_A_idx, :],
                local_vel[:, :, L_A_idx + 1, :],
            ],
            dim=-1,
        )

        # Reshape and concatenate feet information
        Root = root_data
        R_Leg = torch.cat(
            [R_L.view(batch_size, nframes, -1), feet[:, :, 2:]], dim=-1
        )  # (batch_size, nframes, 62)
        L_Leg = torch.cat(
            [L_L.view(batch_size, nframes, -1), feet[:, :, :2]], dim=-1
        )  # (batch_size, nframes, 62)
        Backbone = B.view(batch_size, nframes, -1)  # (batch_size, nframes, 48)
        R_Arm = R_A.view(batch_size, nframes, -1)  # (batch_size, nframes, 48)
        L_Arm = L_A.view(batch_size, nframes, -1)  # (batch_size, nframes, 48)

        return [Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm]

    else:
        raise ValueError(f"Unsupported mode: {mode}")


def parts2whole(parts, mode="t2m", shared_joint_rec_mode="Avg"):
    assert isinstance(parts, list), "Input parts should be a list"

    if mode == "t2m":
        # Parts to whole. (Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm) ==> Combined feature
        Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm = parts

        if len(Root.shape) == 3:  # (bs, nframes, part_repre)
            bs = Root.shape[0]
            nframes = Root.shape[1]
        elif len(Root.shape) == 2:
            bs = None
            nframes = Root.shape[0]
        else:
            raise ValueError("Unsupported Root shape")

        joints_num = 22
        device = Root.device

        rec_root_data = Root[..., :4]  # (batch_size, nframes, 4)
        rec_feet = torch.cat([L_Leg[..., -2:], R_Leg[..., -2:]], dim=-1)  # (batch_size, nframes, 4)

        R_L_idx = torch.tensor([2, 5, 8, 11], dtype=torch.int64, device=device)
        L_L_idx = torch.tensor([1, 4, 7, 10], dtype=torch.int64, device=device)
        B_idx = torch.tensor([3, 6, 9, 12, 15], dtype=torch.int64, device=device)
        R_A_idx = torch.tensor([9, 14, 17, 19, 21], dtype=torch.int64, device=device)
        L_A_idx = torch.tensor([9, 13, 16, 18, 20], dtype=torch.int64, device=device)

        if bs is None:
            R_L = R_Leg[..., :-2].reshape(nframes, 4, -1)  # (nframes, 4, 12)
            L_L = L_Leg[..., :-2].reshape(nframes, 4, -1)  # (nframes, 4, 12)
            B = Backbone.reshape(nframes, 5, -1)  # (nframes, 5, 12)
            R_A = R_Arm.reshape(nframes, 5, -1)  # (nframes, 5, 12)
            L_A = L_Arm.reshape(nframes, 5, -1)  # (nframes, 5, 12)

            rec_ric_data = torch.zeros(
                nframes, joints_num - 1, 3, device=device, dtype=rec_root_data.dtype
            )
            rec_rot_data = torch.zeros(
                nframes, joints_num - 1, 6, device=device, dtype=rec_root_data.dtype
            )
            rec_local_vel = torch.zeros(
                nframes, joints_num, 3, device=device, dtype=rec_root_data.dtype
            )
            rec_local_vel[:, 0, :] = Root[:, 4:]
        else:
            R_L = R_Leg[..., :-2].reshape(bs, nframes, 4, -1)  # (bs, nframes, 4, 12)
            L_L = L_Leg[..., :-2].reshape(bs, nframes, 4, -1)  # (bs, nframes, 4, 12)
            B = Backbone.reshape(bs, nframes, 5, -1)  # (bs, nframes, 5, 12)
            R_A = R_Arm.reshape(bs, nframes, 5, -1)  # (bs, nframes, 5, 12)
            L_A = L_Arm.reshape(bs, nframes, 5, -1)  # (bs, nframes, 5, 12)

            rec_ric_data = torch.zeros(
                bs, nframes, joints_num - 1, 3, device=device, dtype=rec_root_data.dtype
            )
            rec_rot_data = torch.zeros(
                bs, nframes, joints_num - 1, 6, device=device, dtype=rec_root_data.dtype
            )
            rec_local_vel = torch.zeros(
                bs, nframes, joints_num, 3, device=device, dtype=rec_root_data.dtype
            )
            rec_local_vel[..., 0, :] = Root[..., 4:]

    elif mode == "kit":
        Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm = parts

        if len(Root.shape) == 3:  # (bs, nframes, part_repre)
            bs = Root.shape[0]
            nframes = Root.shape[1]
        elif len(Root.shape) == 2:
            bs = None
            nframes = Root.shape[0]
        else:
            raise ValueError("Unsupported Root shape")

        joints_num = 21
        device = Root.device

        rec_root_data = Root[..., :4]  # (batch_size, nframes, 4)
        rec_feet = torch.cat([L_Leg[..., -2:], R_Leg[..., -2:]], dim=-1)  # (batch_size, nframes, 4)

        R_L_idx = torch.tensor([11, 12, 13, 14, 15], dtype=torch.int64, device=device)
        L_L_idx = torch.tensor([16, 17, 18, 19, 20], dtype=torch.int64, device=device)
        B_idx = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device=device)
        R_A_idx = torch.tensor([3, 5, 6, 7], dtype=torch.int64, device=device)
        L_A_idx = torch.tensor([3, 8, 9, 10], dtype=torch.int64, device=device)

        if bs is None:
            R_L = R_Leg[..., :-2].reshape(nframes, 5, -1)  # (nframes, 5, 12)
            L_L = L_Leg[..., :-2].reshape(nframes, 5, -1)  # (nframes, 5, 12)
            B = Backbone.reshape(nframes, 4, -1)  # (nframes, 4, 12)
            R_A = R_Arm.reshape(nframes, 4, -1)  # (nframes, 4, 12)
            L_A = L_Arm.reshape(nframes, 4, -1)  # (nframes, 4, 12)

            rec_ric_data = torch.zeros(
                nframes, joints_num - 1, 3, device=device, dtype=rec_root_data.dtype
            )
            rec_rot_data = torch.zeros(
                nframes, joints_num - 1, 6, device=device, dtype=rec_root_data.dtype
            )
            rec_local_vel = torch.zeros(
                nframes, joints_num, 3, device=device, dtype=rec_root_data.dtype
            )
            rec_local_vel[:, 0, :] = Root[:, 4:]
        else:
            R_L = R_Leg[..., :-2].reshape(bs, nframes, 5, -1)  # (bs, nframes, 5, 12)
            L_L = L_Leg[..., :-2].reshape(bs, nframes, 5, -1)  # (bs, nframes, 5, 12)
            B = Backbone.reshape(bs, nframes, 4, -1)  # (bs, nframes, 4, 12)
            R_A = R_Arm.reshape(bs, nframes, 4, -1)  # (bs, nframes, 4, 12)
            L_A = L_Arm.reshape(bs, nframes, 4, -1)  # (bs, nframes, 4, 12)

            rec_ric_data = torch.zeros(
                bs, nframes, joints_num - 1, 3, device=device, dtype=rec_root_data.dtype
            )
            rec_rot_data = torch.zeros(
                bs, nframes, joints_num - 1, 6, device=device, dtype=rec_root_data.dtype
            )
            rec_local_vel = torch.zeros(
                bs, nframes, joints_num, 3, device=device, dtype=rec_root_data.dtype
            )
            rec_local_vel[..., 0, :] = Root[..., 4:]

    else:
        raise ValueError(f"Unsupported mode: {mode}")

    for part, idx in zip([R_L, L_L, B, R_A, L_A], [R_L_idx, L_L_idx, B_idx, R_A_idx, L_A_idx]):
        adjusted_idx = idx - 1
        rec_ric_data[..., adjusted_idx, :] = part[..., :, :3]
        rec_rot_data[..., adjusted_idx, :] = part[..., :, 3:9]
        rec_local_vel[..., idx, :] = part[..., :, 9:]

    if mode == "t2m" and shared_joint_rec_mode == "Avg":
        idx = 9
        rec_ric_data[..., idx - 1, :] = (L_A[..., 0, :3] + R_A[..., 0, :3] + B[..., 2, :3]) / 3
        rec_rot_data[..., idx - 1, :] = (L_A[..., 0, 3:9] + R_A[..., 0, 3:9] + B[..., 2, 3:9]) / 3
        rec_local_vel[..., idx, :] = (L_A[..., 0, 9:] + R_A[..., 0, 9:] + B[..., 2, 9:]) / 3
    elif mode == "kit":
        idx = 3
        rec_ric_data[..., idx - 1, :] = (L_A[..., 0, :3] + R_A[..., 0, :3] + B[..., 2, :3]) / 3
        rec_rot_data[..., idx - 1, :] = (L_A[..., 0, 3:9] + R_A[..., 0, 3:9] + B[..., 2, 3:9]) / 3
        rec_local_vel[..., idx, :] = (L_A[..., 0, 9:] + R_A[..., 0, 9:] + B[..., 2, 9:]) / 3
    else:
        if mode == "t2m":
            raise ValueError(
                f"Unsupported shared_joint_rec_mode for mode 't2m': {shared_joint_rec_mode}"
            )
        elif mode == "kit":
            raise ValueError(
                f"Unsupported shared_joint_rec_mode for mode 'kit': {shared_joint_rec_mode}"
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    if bs is None:
        rec_ric_data = rec_ric_data.reshape(nframes, -1)
        rec_rot_data = rec_rot_data.reshape(nframes, -1)
        rec_local_vel = rec_local_vel.reshape(nframes, -1)
        rec_data = torch.cat(
            [rec_root_data, rec_ric_data, rec_rot_data, rec_local_vel, rec_feet], dim=1
        )  # (nframes, 263)
    else:
        rec_ric_data = rec_ric_data.reshape(bs, nframes, -1)
        rec_rot_data = rec_rot_data.reshape(bs, nframes, -1)
        rec_local_vel = rec_local_vel.reshape(bs, nframes, -1)
        rec_data = torch.cat(
            [rec_root_data, rec_ric_data, rec_rot_data, rec_local_vel, rec_feet], dim=2
        )  # (bs, nframes, 263)

    return rec_data


def get_each_part_vel(parts, mode="kit"):
    assert isinstance(parts, list), "Input parts should be a list"

    if mode == "t2m":
        # Extract each part's velocity from parts representation
        Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm = parts

        if len(Root.shape) == 3:  # (bs, nframes, part_repre)
            bs = Root.shape[0]
            nframes = Root.shape[1]
        elif len(Root.shape) == 2:  # (nframes, part_repre)
            bs = None
            nframes = Root.shape[0]
        else:
            raise ValueError("Unsupported Root shape")

        Root_vel = Root[..., 4:]
        if bs is None:
            R_L = R_Leg[:, :-2].reshape(nframes, 4, -1)  # (nframes, 4, 12)
            L_L = L_Leg[:, :-2].reshape(nframes, 4, -1)  # (nframes, 4, 12)
            B = Backbone.reshape(nframes, 5, -1)  # (nframes, 5, 12)
            R_A = R_Arm.reshape(nframes, 5, -1)  # (nframes, 5, 12)
            L_A = L_Arm.reshape(nframes, 5, -1)  # (nframes, 5, 12)

            R_Leg_vel = R_L[:, :, 9:].reshape(nframes, -1)
            L_Leg_vel = L_L[:, :, 9:].reshape(nframes, -1)
            Backbone_vel = B[:, :, 9:].reshape(nframes, -1)
            R_Arm_vel = R_A[:, :, 9:].reshape(nframes, -1)
            L_Arm_vel = L_A[:, :, 9:].reshape(nframes, -1)

        else:
            R_L = R_Leg[:, :, :-2].reshape(bs, nframes, 4, -1)  # (bs, nframes, 4, 12)
            L_L = L_Leg[:, :, :-2].reshape(bs, nframes, 4, -1)  # (bs, nframes, 4, 12)
            B = Backbone.reshape(bs, nframes, 5, -1)  # (bs, nframes, 5, 12)
            R_A = R_Arm.reshape(bs, nframes, 5, -1)  # (bs, nframes, 5, 12)
            L_A = L_Arm.reshape(bs, nframes, 5, -1)  # (bs, nframes, 5, 12)

            R_Leg_vel = R_L[:, :, :, 9:].reshape(bs, nframes, -1)  # (bs, nframes, vel_dim)
            L_Leg_vel = L_L[:, :, :, 9:].reshape(bs, nframes, -1)
            Backbone_vel = B[:, :, :, 9:].reshape(bs, nframes, -1)
            R_Arm_vel = R_A[:, :, :, 9:].reshape(bs, nframes, -1)
            L_Arm_vel = L_A[:, :, :, 9:].reshape(bs, nframes, -1)

        parts_vel_list = [
            Root_vel,
            R_Leg_vel,
            L_Leg_vel,
            Backbone_vel,
            R_Arm_vel,
            L_Arm_vel,
        ]

    elif mode == "kit":
        # Extract each part's velocity from parts representation
        Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm = parts

        if len(Root.shape) == 3:  # (bs, nframes, part_repre)
            bs = Root.shape[0]
            nframes = Root.shape[1]
        elif len(Root.shape) == 2:  # (nframes, part_repre)
            bs = None
            nframes = Root.shape[0]
        else:
            raise ValueError("Unsupported Root shape")

        Root_vel = Root[..., 4:]
        if bs is None:
            R_L = R_Leg[:, :-2].reshape(nframes, 5, -1)  # (nframes, 5, 12)
            L_L = L_Leg[:, :-2].reshape(nframes, 5, -1)  # (nframes, 5, 12)
            B = Backbone.reshape(nframes, 4, -1)  # (nframes, 4, 12)
            R_A = R_Arm.reshape(nframes, 4, -1)  # (nframes, 4, 12)
            L_A = L_Arm.reshape(nframes, 4, -1)  # (nframes, 4, 12)

            R_Leg_vel = R_L[:, :, 9:].reshape(nframes, -1)
            L_Leg_vel = L_L[:, :, 9:].reshape(nframes, -1)
            Backbone_vel = B[:, :, 9:].reshape(nframes, -1)
            R_Arm_vel = R_A[:, :, 9:].reshape(nframes, -1)
            L_Arm_vel = L_A[:, :, 9:].reshape(nframes, -1)

        else:
            R_L = R_Leg[:, :, :-2].reshape(bs, nframes, 5, -1)  # (bs, nframes, 5, 12)
            L_L = L_Leg[:, :, :-2].reshape(bs, nframes, 5, -1)  # (bs, nframes, 5, 12)
            B = Backbone.reshape(bs, nframes, 4, -1)  # (bs, nframes, 4, 12)
            R_A = R_Arm.reshape(bs, nframes, 4, -1)  # (bs, nframes, 4, 12)
            L_A = L_Arm.reshape(bs, nframes, 4, -1)  # (bs, nframes, 4, 12)

            R_Leg_vel = R_L[:, :, :, 9:].reshape(bs, nframes, -1)  # (bs, nframes, vel_dim)
            L_Leg_vel = L_L[:, :, :, 9:].reshape(bs, nframes, -1)
            Backbone_vel = B[:, :, :, 9:].reshape(bs, nframes, -1)
            R_Arm_vel = R_A[:, :, :, 9:].reshape(bs, nframes, -1)
            L_Arm_vel = L_A[:, :, :, 9:].reshape(bs, nframes, -1)

        parts_vel_list = [
            Root_vel,
            R_Leg_vel,
            L_Leg_vel,
            Backbone_vel,
            R_Arm_vel,
            L_Arm_vel,
        ]

    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return parts_vel_list  # [Root_vel, R_Leg_vel, L_Leg_vel, Backbone_vel, R_Arm_vel, L_Arm_vel]
